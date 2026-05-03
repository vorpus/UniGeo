"""Persistent Wan2.2 + helper renderers for the web UI.

- WanWorker holds the loaded WanVideoPipeline + LoRA in process memory across
  many generate calls (saves ~60s of model load per request).
- render_trajectory_mp4() builds the point-cloud preview video from a single
  end-camera transform (linearly interpolated from origin over num_frames).
"""

import json
import os
import sys
from typing import Callable, Optional

import cv2
import imageio
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from scipy.spatial.transform import Rotation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "DiffSynth-Studio"))
sys.path.insert(0, os.path.join(REPO, "vggt"))

from diffsynth.core.data.operators import (  # noqa: E402
    ImageCropAndResize,
    LoadImage,
    LoadVideo,
    RouteByExtensionName,
    RouteByType,
    ToAbsolutePath,
    ToList,
)
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline  # noqa: E402

NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的"
    "脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)


# ---------- camera trajectory rendering ----------

def _build_estimate_rel(x, y, z, phi_deg, theta_deg):
    rot = Rotation.from_euler("xyz", [theta_deg, phi_deg, 0.0], degrees=True).as_matrix()
    M = np.eye(4)
    M[:3, :3] = rot
    M[:3, 3] = [x, y, z]
    return M


def render_trajectory_mp4(
    image_path: str,
    points_xyz: np.ndarray,  # (N,3) float32 in VGGT world frame
    points_rgb: np.ndarray,  # (N,3) float in [0,1]
    image_size_hw: tuple,
    focals: tuple,
    principal: tuple,
    end_xyz_phi_theta: tuple,  # (x, y, z, phi_deg, theta_deg)
    out_mp4: str,
    num_frames: int = 81,  # LoadVideo hardcodes np.linspace(0, 80, ...) so source needs 81 frames
    fps: int = 30,
    device: str = "cuda",
):
    from pytorch3d.renderer import (
        AlphaCompositor,
        PerspectiveCameras,
        PointsRasterizationSettings,
        PointsRasterizer,
        PointsRenderer,
    )
    from pytorch3d.structures import Pointclouds

    H, W = image_size_hw
    raw_image = np.array(Image.open(image_path).convert("RGB").resize((W, H))) / 255.0

    pts = torch.from_numpy(points_xyz).float().to(device)
    col = torch.from_numpy(points_rgb).float().to(device)
    if col.max() > 1.5:
        col = col / 255.0

    focals_t = torch.tensor([focals], dtype=torch.float32, device=device)
    principal_t = torch.tensor([principal], dtype=torch.float32, device=device)

    x, y, z, phi, theta = end_xyz_phi_theta

    rendered = []
    for i in range(num_frames):
        a = i / max(num_frames - 1, 1)
        M = _build_estimate_rel(x * a, y * a, z * a, phi * a, theta * a)
        M = torch.from_numpy(M).float().to(device)
        rel = M.unsqueeze(0)
        R, T = rel[:, :3, :3], rel[:, :3, 3:]
        R = torch.stack([-R[:, :, 0], -R[:, :, 1], R[:, :, 2]], 2)
        c2w = torch.cat([R, T], 2)
        bottom = torch.tensor([[[0, 0, 0, 1]]], device=device).repeat(c2w.shape[0], 1, 1)
        w2c = torch.linalg.inv(torch.cat([c2w, bottom], 1))
        R_new = w2c[:, :3, :3].permute(0, 2, 1)
        T_new = w2c[:, :3, 3]

        cameras = PerspectiveCameras(
            focal_length=focals_t,
            principal_point=principal_t,
            in_ndc=False,
            image_size=((H, W),),
            R=R_new,
            T=T_new,
            device=device,
        )
        raster = PointsRasterizationSettings(image_size=(H, W), radius=0.01, points_per_pixel=10, bin_size=0)
        renderer = PointsRenderer(rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster), compositor=AlphaCompositor())
        pc = Pointclouds(points=[pts], features=[col]).extend(1)
        out = renderer(pc)[0].detach().cpu().numpy()
        out = (out * 255).clip(0, 255).astype(np.uint8)
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        elif out.shape[-1] == 4:
            out = out[..., :3]
        rendered.append(out)

    rendered[0] = (raw_image * 255).clip(0, 255).astype(np.uint8)
    imageio.mimwrite(out_mp4, rendered, fps=fps, codec="libx264", quality=8)


# ---------- persistent Wan worker ----------

class WanWorker:
    HEIGHT = 704
    WIDTH = 1248
    NUM_FRAMES = 25
    NUM_INFERENCE_STEPS = 50

    def __init__(self, wan_dir: str, lora_path: str, wan_config_path: str):
        self.wan_dir = wan_dir
        self.lora_path = lora_path
        self.wan_config_path = wan_config_path
        self.pipe: Optional[WanVideoPipeline] = None

    def load(self):
        if self.pipe is not None:
            return
        vram_config = {
            "offload_dtype": torch.bfloat16,
            "offload_device": "cpu",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cuda:0",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda:0",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda:0",
        }
        wan_paths = [
            os.path.join(self.wan_dir, f"diffusion_pytorch_model-0000{i}-of-00003.safetensors")
            for i in (1, 2, 3)
        ]
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(path=os.path.join(self.wan_dir, "models_t5_umt5-xxl-enc-bf16.pth"), **vram_config),
                ModelConfig(path=os.path.join(self.wan_dir, "Wan2.2_VAE.pth"), **vram_config),
            ],
            tokenizer_config=ModelConfig(path=os.path.join(self.wan_dir, "google/umt5-xxl/")),
            wan_paths=wan_paths,
            wan_config_path=self.wan_config_path,
            vram_limit=20,
        )

        ckpt = load_file(self.lora_path)
        lora_sd = {k: v for k, v in ckpt.items() if ".lora_" in k}
        adapter_sd = {k: v for k, v in ckpt.items() if "i2v_adapter" in k}
        self.pipe.load_lora(self.pipe.dit, state_dict=lora_sd, alpha=1)
        self.pipe.dit.load_state_dict(adapter_sd, strict=False)
        self.pipe.to("cuda")
        self.pipe.to(dtype=torch.bfloat16)

    def infer(
        self,
        src_image_path: str,
        src_video_path: str,
        out_png_path: str,
        out_mp4_path: Optional[str] = None,
        prompt: str = "Ensure the consistency of the video",
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ):
        assert self.pipe is not None, "call load() first"

        op = RouteByType(operator_map=[(
            str,
            ToAbsolutePath("") >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(self.HEIGHT, self.WIDTH, self.HEIGHT * self.WIDTH, 16, 16) >> ToList()),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(self.NUM_FRAMES, 4, 1, frame_processor=ImageCropAndResize(self.HEIGHT, self.WIDTH, self.HEIGHT * self.WIDTH, 16, 16))),
            ]),
        )])
        src_video = op(src_video_path)
        src_image_list = op(src_image_path)

        if progress_cb:
            progress_cb("infer", 0.0)

        # The pipeline iterates timesteps via `progress_bar_cmd(scheduler.timesteps)`.
        # Pass a callable that yields each item and pings the websocket per step.
        def progress_iter(it):
            it = list(it)
            total = len(it)
            for i, x in enumerate(it):
                yield x
                if progress_cb:
                    progress_cb("infer", (i + 1) / total)

        video = self.pipe(
            prompt=prompt,
            negative_prompt=NEG_PROMPT,
            src_video=src_video,
            input_image=src_image_list[0],
            height=self.HEIGHT,
            width=self.WIDTH,
            cfg_scale=5.0,
            num_frames=self.NUM_FRAMES + 4,
            num_inference_steps=self.NUM_INFERENCE_STEPS,
            seed=0,
            tiled=True,
            progress_bar_cmd=progress_iter,
        )

        frames = list(video)
        last_np = np.array(frames[-1])
        os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
        imageio.imwrite(out_png_path, last_np)
        if out_mp4_path:
            os.makedirs(os.path.dirname(out_mp4_path), exist_ok=True)
            imageio.mimwrite(
                out_mp4_path,
                [np.array(f) for f in frames],
                fps=15,
                codec="libx264",
                quality=8,
            )
        if progress_cb:
            progress_cb("infer", 1.0)
