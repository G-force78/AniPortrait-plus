import gradio as gr
import os
import shutil
import ffmpeg
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import torch
import onnxruntime
import onnxruntime as ort
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from scipy.interpolate import interp1d
from tqdm import tqdm
from face_enhancer import (
    get_available_enhancer_names,
    load_face_enhancer_model,
    cv2_interpolations,
)
import tempfile
import uuid
import os
import cv2
import ffmpeg
from face_enhancer import load_face_enhancer_model

from tqdm import tqdm


from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid

from src.audio_models.model import Audio2MeshModel
from src.utils.audio_util import prepare_audio_feature
from src.utils.mp_utils import LMKExtractor
from src.utils.draw_util import FaceMeshVisualizer
from src.utils.pose_util import (
    project_points,
    project_points_with_trans,
    matrix_to_euler_and_translation,
    euler_and_translation_to_matrix,
)
from src.utils.util import crop_face
from scripts.vid2vid import smooth_pose_seq
from src.utils.frame_interpolation import (
    init_frame_interpolation_model,
    batch_images_interpolation_tool,
)


## ------------------------------ FACE ENHANCEMENT ------------------------------
# to be continued
PROVIDER = ["CUDAExecutionProvider"]


available_providers = onnxruntime.get_available_providers()
if "CUDAExecutionProvider" in available_providers:
    print("\n********** Running on CUDA **********\n")
# PROVIDER = ["CUDAExecutionProvider", "CPUExecutionProvider"]


device = "cuda"
EMPTY_CACHE = lambda: torch.cuda.empty_cache() if device == "cuda" else None


config = OmegaConf.load(
    "/content/AniPortrait-plus/configs/prompts/animation_audio.yaml"
)
if config.weight_dtype == "fp16":
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32

audio_infer_config = OmegaConf.load(config.audio_inference_config)
# prepare model
a2m_model = Audio2MeshModel(audio_infer_config["a2m_model"])
a2m_model.load_state_dict(
    torch.load(audio_infer_config["pretrained_model"]["a2m_ckpt"], map_location="cpu"),
    strict=False,
)
a2m_model.cuda().eval()

vae = AutoencoderKL.from_pretrained(
    config.pretrained_vae_path,
).to("cuda", dtype=weight_dtype)

reference_unet = UNet2DConditionModel.from_pretrained(
    config.pretrained_base_model_path,
    subfolder="unet",
).to(dtype=weight_dtype, device="cuda")

inference_config_path = config.inference_config
infer_config = OmegaConf.load(inference_config_path)
denoising_unet = UNet3DConditionModel.from_pretrained_2d(
    config.pretrained_base_model_path,
    config.motion_module_path,
    subfolder="unet",
    unet_additional_kwargs=infer_config.unet_additional_kwargs,
).to(dtype=weight_dtype, device="cuda")

pose_guider = PoseGuider(noise_latent_channels=320, use_ca=True).to(
    device="cuda", dtype=weight_dtype
)  # not use cross attention

image_enc = CLIPVisionModelWithProjection.from_pretrained(config.image_encoder_path).to(
    dtype=weight_dtype, device="cuda"
)

sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
scheduler = DDIMScheduler(**sched_kwargs)

# load pretrained weights
denoising_unet.load_state_dict(
    torch.load(config.denoising_unet_path, map_location="cpu"),
    strict=False,
)
reference_unet.load_state_dict(
    torch.load(config.reference_unet_path, map_location="cpu"),
)
pose_guider.load_state_dict(
    torch.load(config.pose_guider_path, map_location="cpu"),
)

pipe = Pose2VideoPipeline(
    vae=vae,
    image_encoder=image_enc,
    reference_unet=reference_unet,
    denoising_unet=denoising_unet,
    pose_guider=pose_guider,
    scheduler=scheduler,
)
pipe = pipe.to("cuda", dtype=weight_dtype)

frame_inter_model = init_frame_interpolation_model()

# def upscale_video_with_face_enhancer(video_path, method):
# Load the face enhancement model
# face_enhancer = load_face_enhancer_model(method)

# Upscale the video using the face enhancement model
# input_video = ffmpeg.input(video_path)
# output_video = ffmpeg.output(input_video.video, f"{os.path.splitext(video_path)[0]}_upscaled.mp4", vcodec='libx264', r=30)
# ffmpeg.run(output_video)

# return f"{os.path.splitext(video_path)[0]}_upscaled.mp4"


####UPSCALE VIDEOS#####


def upscale_video_with_face_enhancer(video_path, method, device, batch_size=12):
    # Load the face enhancement model
    face_enhancer, face_enhancer_runner = load_face_enhancer_model(
        method, device="cuda:0"
    )

    # Create a temporary directory to store the frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract frames from the input video
        frame_paths = []
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        pbar = tqdm(total=total_frames, unit="frames")  # Create a progress bar

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_idx += 1
            pbar.update(1)  # Update the progress bar

        cap.release()
        pbar.close()  # Close the progress bar

        # Upscale the frames in batches
        upscaled_frames = []
        for i in tqdm(
            range(0, len(frame_paths), batch_size),
            unit="batches",
            desc="Upscaling frames",
        ):
            batch_frame_paths = frame_paths[i : i + batch_size]
            batch_frames = [cv2.imread(path) for path in batch_frame_paths]
            if face_enhancer is None:
                batch_upscaled_frames = [
                    face_enhancer_runner(frame, None) for frame in batch_frames
                ]
            else:
                batch_upscaled_frames = [
                    face_enhancer_runner(frame, face_enhancer) for frame in batch_frames
                ]
            upscaled_frames.extend(batch_upscaled_frames)

        # Create the output video from the upscaled frames
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            height, width, _ = upscaled_frames[0].shape
            out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))
            for frame in upscaled_frames:
                out.write(frame)
            out.release()

            # Extract the audio from the original video
            audio_output = f"{temp_dir}/audio.aac"
            try:
                ffmpeg.input(video_path).output(audio_output, acodec="copy").run()
            except:
                print(f"Failed to extract audio from {video_path}")
                audio_output = None

            # Generate the final video path with "_upscaled" at the end
            unique_id = str(uuid.uuid4())
            final_video_path = (
                f"{os.path.splitext(video_path)[0]}_{unique_id}_upscaled.mp4"
            )

            # Merge the upscaled video and the audio
            stream = ffmpeg.input(temp_file.name)
            if audio_output:
                audio = ffmpeg.input(audio_output)
                ffmpeg.output(
                    stream.video,
                    audio.audio,
                    final_video_path,
                    vcodec="copy",
                    acodec="aac",
                    shortest=None,
                ).run()
            else:
                ffmpeg.output(stream.video, final_video_path, vcodec="copy").run()

    # Return the path to the final video
    return final_video_path


def get_headpose_temp(input_video):
    lmk_extractor = LMKExtractor()
    cap = cv2.VideoCapture(input_video)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    trans_mat_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = lmk_extractor(frame)
        trans_mat_list.append(result["trans_mat"].astype(np.float32))
    cap.release()

    trans_mat_arr = np.array(trans_mat_list)

    # compute delta pose
    trans_mat_inv_frame_0 = np.linalg.inv(trans_mat_arr[0])
    pose_arr = np.zeros([trans_mat_arr.shape[0], 6])

    for i in range(pose_arr.shape[0]):
        pose_mat = trans_mat_inv_frame_0 @ trans_mat_arr[i]
        euler_angles, translation_vector = matrix_to_euler_and_translation(pose_mat)
        pose_arr[i, :3] = euler_angles
        pose_arr[i, 3:6] = translation_vector

    # interpolate to 30 fps
    new_fps = 30
    old_time = np.linspace(0, total_frames / fps, total_frames)
    new_time = np.linspace(0, total_frames / fps, int(total_frames * new_fps / fps))

    pose_arr_interp = np.zeros((len(new_time), 6))
    for i in range(6):
        interp_func = interp1d(old_time, pose_arr[:, i])
        pose_arr_interp[:, i] = interp_func(new_time)

    pose_arr_smooth = smooth_pose_seq(pose_arr_interp)

    return pose_arr_smooth


def audio2video(
    input_audio,
    ref_img,
    headpose_video=None,
    size=512,
    steps=10,
    length=0,
    seed=42,
    acc_flag=True,
):
    fps = 30
    cfg = 3.5
    fi_step = 3 if acc_flag else 1

    generator = torch.manual_seed(seed)

    lmk_extractor = LMKExtractor()
    vis = FaceMeshVisualizer()

    width, height = size, size

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{seed}-{size}x{size}"

    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    while os.path.exists(save_dir):
        save_dir = Path(
            f"output/{date_str}/{save_dir_name}_{np.random.randint(10000):04d}"
        )
    save_dir.mkdir(exist_ok=True, parents=True)

    ref_image_np = cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)
    ref_image_np = crop_face(ref_image_np, lmk_extractor)
    if ref_image_np is None:
        return None, Image.fromarray(ref_img)

    ref_image_np = cv2.resize(ref_image_np, (size, size))
    ref_image_pil = Image.fromarray(cv2.cvtColor(ref_image_np, cv2.COLOR_BGR2RGB))

    face_result = lmk_extractor(ref_image_np)
    if face_result is None:
        return None, ref_image_pil

    lmks = face_result["lmks"].astype(np.float32)
    ref_pose = vis.draw_landmarks(
        (ref_image_np.shape[1], ref_image_np.shape[0]), lmks, normed=True
    )

    sample = prepare_audio_feature(
        input_audio, wav2vec_model_path=audio_infer_config["a2m_model"]["model_path"]
    )
    sample["audio_feature"] = torch.from_numpy(sample["audio_feature"]).float().cuda()
    sample["audio_feature"] = sample["audio_feature"].unsqueeze(0)

    # inference
    pred = a2m_model.infer(sample["audio_feature"], sample["seq_len"])
    pred = pred.squeeze().detach().cpu().numpy()
    pred = pred.reshape(pred.shape[0], -1, 3)
    pred = pred + face_result["lmks3d"]

    if headpose_video is not None:
        pose_seq = get_headpose_temp(headpose_video)
    else:
        pose_seq = np.load(config["pose_temp"])
    mirrored_pose_seq = np.concatenate((pose_seq, pose_seq[-2:0:-1]), axis=0)
    cycled_pose_seq = np.tile(
        mirrored_pose_seq, (sample["seq_len"] // len(mirrored_pose_seq) + 1, 1)
    )[: sample["seq_len"]]

    # project 3D mesh to 2D landmark
    projected_vertices = project_points(
        pred, face_result["trans_mat"], cycled_pose_seq, [height, width]
    )

    pose_images = []
    for i, verts in enumerate(projected_vertices):
        lmk_img = vis.draw_landmarks((width, height), verts, normed=False)
        pose_images.append(lmk_img)

    pose_list = []
    args_L = len(pose_images) if length == 0 or length > len(pose_images) else length
    for pose_image_np in pose_images[:args_L:fi_step]:
        pose_image_np = cv2.resize(pose_image_np, (width, height))
        pose_list.append(pose_image_np)

    pose_list = np.array(pose_list)

    video_length = len(pose_list)

    video = pipe(
        ref_image_pil,
        pose_list,
        ref_pose,
        width,
        height,
        video_length,
        steps,
        cfg,
        generator=generator,
    ).videos

    if acc_flag:
        video = batch_images_interpolation_tool(
            video, frame_inter_model, inter_frames=fi_step - 1
        )

    save_path = f"{save_dir}/{size}x{size}_{time_str}_noaudio.mp4"
    save_videos_grid(
        video,
        save_path,
        n_rows=1,
        fps=fps,
    )

    stream = ffmpeg.input(save_path)
    audio = ffmpeg.input(input_audio)
    ffmpeg.output(
        stream.video,
        audio.audio,
        save_path.replace("_noaudio.mp4", ".mp4"),
        vcodec="copy",
        acodec="aac",
        shortest=None,
    ).run()
    os.remove(save_path)

    return save_path.replace("_noaudio.mp4", ".mp4"), ref_image_pil


def video2video(
    ref_img, source_video, size=512, steps=10, length=0, seed=42, acc_flag=True
):
    cfg = 3.5
    fi_step = 3 if acc_flag else 1

    generator = torch.manual_seed(seed)

    lmk_extractor = LMKExtractor()
    vis = FaceMeshVisualizer()

    width, height = size, size

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{seed}-{size}x{size}"

    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    while os.path.exists(save_dir):
        save_dir = Path(
            f"output/{date_str}/{save_dir_name}_{np.random.randint(10000):04d}"
        )
    save_dir.mkdir(exist_ok=True, parents=True)

    ref_image_np = cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)
    ref_image_np = crop_face(ref_image_np, lmk_extractor)
    if ref_image_np is None:
        return None, Image.fromarray(ref_img)

    ref_image_np = cv2.resize(ref_image_np, (size, size))
    ref_image_pil = Image.fromarray(cv2.cvtColor(ref_image_np, cv2.COLOR_BGR2RGB))

    face_result = lmk_extractor(ref_image_np)
    if face_result is None:
        return None, ref_image_pil

    lmks = face_result["lmks"].astype(np.float32)
    ref_pose = vis.draw_landmarks(
        (ref_image_np.shape[1], ref_image_np.shape[0]), lmks, normed=True
    )

    source_images = read_frames(source_video)
    src_fps = get_fps(source_video)

    step = 1
    if src_fps == 60:
        src_fps = 30
        step = 2

    pose_trans_list = []
    verts_list = []
    bs_list = []
    args_L = (
        len(source_images)
        if length == 0 or length * step > len(source_images)
        else length * step
    )
    for src_image_pil in source_images[: args_L : step * fi_step]:
        src_img_np = cv2.cvtColor(np.array(src_image_pil), cv2.COLOR_RGB2BGR)
        frame_height, frame_width, _ = src_img_np.shape
        src_img_result = lmk_extractor(src_img_np)
        if src_img_result is None:
            break
        pose_trans_list.append(src_img_result["trans_mat"])
        verts_list.append(src_img_result["lmks3d"])
        bs_list.append(src_img_result["bs"])

    trans_mat_arr = np.array(pose_trans_list)
    verts_arr = np.array(verts_list)
    bs_arr = np.array(bs_list)
    min_bs_idx = np.argmin(bs_arr.sum(1))

    # compute delta pose
    pose_arr = np.zeros([trans_mat_arr.shape[0], 6])

    for i in range(pose_arr.shape[0]):
        euler_angles, translation_vector = matrix_to_euler_and_translation(
            trans_mat_arr[i]
        )  # real pose of source
        pose_arr[i, :3] = euler_angles
        pose_arr[i, 3:6] = translation_vector

    init_tran_vec = face_result["trans_mat"][:3, 3]  # init translation of tgt
    pose_arr[:, 3:6] = (
        pose_arr[:, 3:6] - pose_arr[0, 3:6] + init_tran_vec
    )  # (relative translation of source) + (init translation of tgt)

    pose_arr_smooth = smooth_pose_seq(pose_arr, window_size=3)
    pose_mat_smooth = [
        euler_and_translation_to_matrix(pose_arr_smooth[i][:3], pose_arr_smooth[i][3:6])
        for i in range(pose_arr_smooth.shape[0])
    ]
    pose_mat_smooth = np.array(pose_mat_smooth)

    # face retarget
    verts_arr = verts_arr - verts_arr[min_bs_idx] + face_result["lmks3d"]
    # project 3D mesh to 2D landmark
    projected_vertices = project_points_with_trans(
        verts_arr, pose_mat_smooth, [frame_height, frame_width]
    )

    pose_list = []
    for i, verts in enumerate(projected_vertices):
        lmk_img = vis.draw_landmarks((frame_width, frame_height), verts, normed=False)
        pose_image_np = cv2.resize(lmk_img, (width, height))
        pose_list.append(pose_image_np)

    pose_list = np.array(pose_list)

    video_length = len(pose_list)

    video = pipe(
        ref_image_pil,
        pose_list,
        ref_pose,
        width,
        height,
        video_length,
        steps,
        cfg,
        generator=generator,
    ).videos

    if acc_flag:
        video = batch_images_interpolation_tool(
            video, frame_inter_model, inter_frames=fi_step - 1
        )

    save_path = f"{save_dir}/{size}x{size}_{time_str}_noaudio.mp4"
    save_videos_grid(
        video,
        save_path,
        n_rows=1,
        fps=src_fps,
    )

    audio_output = f"{save_dir}/audio_from_video.aac"
    # extract audio
    try:
        ffmpeg.input(source_video).output(audio_output, acodec="copy").run()
        # merge audio and video
        stream = ffmpeg.input(save_path)
        audio = ffmpeg.input(audio_output)
        ffmpeg.output(
            stream.video,
            audio.audio,
            save_path.replace("_noaudio.mp4", ".mp4"),
            vcodec="copy",
            acodec="aac",
            shortest=None,
        ).run()

        os.remove(save_path)
        os.remove(audio_output)
    except:
        shutil.move(save_path, save_path.replace("_noaudio.mp4", ".mp4"))

    return save_path.replace("_noaudio.mp4", ".mp4"), ref_image_pil


################# GUI ################

title = r"""
<h1>AniPortrait</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/Zejun-Yang/AniPortrait' target='_blank'><b>AniPortrait: Audio-Driven Synthesis of Photorealistic Portrait Animations</b></a>.<br>
"""

with gr.Blocks() as demo:

    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Tab("Audio2video"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    a2v_input_audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        editable=True,
                        label="Input audio",
                        interactive=True,
                    )
                    a2v_ref_img = gr.Image(
                        label="Upload reference image", sources="upload"
                    )
                    a2v_headpose_video = gr.Video(
                        label="Option: upload head pose reference video",
                        sources="upload",
                    )

                with gr.Row():
                    a2v_size_slider = gr.Slider(
                        minimum=256,
                        maximum=768,
                        step=8,
                        value=512,
                        label="Video size (-W & -H)",
                    )
                    a2v_step_slider = gr.Slider(
                        minimum=5, maximum=50, step=1, value=10, label="Steps (--steps)"
                    )

                with gr.Row():
                    a2v_length = gr.Slider(
                        minimum=0,
                        maximum=9999,
                        step=1,
                        value=0,
                        label="Length (-L) (Set to 0 to automatically calculate length)",
                    )
                    a2v_seed = gr.Number(value=42, label="Seed (--seed)")

                with gr.Row():
                    a2v_acc_flag = gr.Checkbox(value=True, label="Accelerate")
                    a2v_botton = gr.Button("Generate", variant="primary")
            a2v_output_video = gr.PlayableVideo(label="Result", interactive=False)

        # gr.Examples(
        # @  examples=[
        #      ["configs/inference/audio/lyl.wav", "configs/inference/ref_images/Aragaki.png", None],
        #      ["configs/inference/audio/lyl.wav", "configs/inference/ref_images/solo.png", None],
        #     ["configs/inference/audio/lyl.wav", "configs/inference/ref_images/lyl.png", "configs/inference/head_pose_temp/pose_ref_video.mp4"],
        #     ],
        #  inputs=[a2v_input_audio, a2v_ref_img, a2v_headpose_video],
        # )

    with gr.Tab("Video2video"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    v2v_ref_img = gr.Image(
                        label="Upload reference image", sources="upload"
                    )
                    v2v_source_video = gr.Video(
                        label="Upload source video", sources="upload"
                    )

                with gr.Row():
                    v2v_size_slider = gr.Slider(
                        minimum=256,
                        maximum=768,
                        step=8,
                        value=512,
                        label="Video size (-W & -H)",
                    )
                    v2v_step_slider = gr.Slider(
                        minimum=5, maximum=50, step=1, value=10, label="Steps (--steps)"
                    )

                with gr.Row():
                    v2v_length = gr.Slider(
                        minimum=0,
                        maximum=9999,
                        step=1,
                        value=0,
                        label="Length (-L) (Set to 0 to automatically calculate length)",
                    )
                    v2v_seed = gr.Number(value=42, label="Seed (--seed)")

                with gr.Row():
                    v2v_acc_flag = gr.Checkbox(value=True, label="Accelerate")
                    v2v_botton = gr.Button("Generate", variant="primary")
            v2v_output_video = gr.PlayableVideo(label="Result", interactive=False)

        # gr.Examples(
        #   examples=[
        #       ["configs/inference/ref_images/Aragaki.png", "configs/inference/video/Aragaki_song.mp4"],
        #      ["configs/inference/ref_images/solo.png", "configs/inference/video/Aragaki_song.mp4"],
        #      ["configs/inference/ref_images/lyl.png", "configs/inference/head_pose_temp/pose_ref_video.mp4"],
        #      ],
        #  inputs=[v2v_ref_img, v2v_source_video, a2v_headpose_video],
        # )
    with gr.Tab("Video Upscale"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    video_path_textbox = gr.Textbox(label="Enter video path:")
                    upscale_method = gr.Dropdown(
                        get_available_enhancer_names(),
                        label="Upscale method - codeformer VERY slow",
                        value="REAL-ESRGAN 4x",
                    )
        upscale_botton = gr.Button("Upscale", variant="primary")
        upscale_output_video = gr.PlayableVideo(
            label="Upscaled video", interactive=False
        )

    a2v_botton.click(
        fn=audio2video,
        inputs=[
            a2v_input_audio,
            a2v_ref_img,
            a2v_headpose_video,
            a2v_size_slider,
            a2v_step_slider,
            a2v_length,
            a2v_seed,
            a2v_acc_flag,
        ],
        outputs=[a2v_output_video, a2v_ref_img],
    )
    v2v_botton.click(
        fn=video2video,
        inputs=[
            v2v_ref_img,
            v2v_source_video,
            v2v_size_slider,
            v2v_step_slider,
            v2v_length,
            v2v_seed,
            v2v_acc_flag,
        ],
        outputs=[v2v_output_video, v2v_ref_img],
    )

    upscale_botton.click(
        fn=upscale_video_with_face_enhancer,
        inputs=[video_path_textbox, upscale_method],
        outputs=[upscale_output_video],
    )


demo.launch(share=True)
