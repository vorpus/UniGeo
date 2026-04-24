import argparse
import torch
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
import torch, os, imageio
import numpy as np
from diffsynth.core.data.operators import *
from safetensors.torch import load_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--wan_model_dir", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--wan_config_path", type=str, required=True)
    return parser.parse_args()


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
    ):
        self.base_path = base_path
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.data = []
        self.load_metadata()
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    
    def load_metadata(self):

        src_dir = os.path.join(self.base_path, "point_cloud")
        tgt_dir = self.base_path

        video_exts = ".mp4"

        for fname in os.listdir(src_dir):

            if not fname.lower().endswith(video_exts):
                continue

            src_path = os.path.join(src_dir, fname)
            tgt_path = os.path.join(tgt_dir, fname).replace(".mp4", ".png")

            self.data.append({
                "src_video": src_path,
                "src_image": tgt_path,
                "prompt": "Ensure the consistency of the video",
                "path": src_path
            })

        print(f"Found {len(self.data)} video pairs")


    def __getitem__(self, data_id):

        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':

    args = get_args()

    num_frames = 25

    wan_paths = [
        os.path.join(args.wan_model_dir, "diffusion_pytorch_model-00001-of-00003.safetensors"),
        os.path.join(args.wan_model_dir, "diffusion_pytorch_model-00002-of-00003.safetensors"),
        os.path.join(args.wan_model_dir, "diffusion_pytorch_model-00003-of-00003.safetensors"),
    ]

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=os.path.join(args.wan_model_dir, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=os.path.join(args.wan_model_dir, "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=os.path.join(args.wan_model_dir, "google/umt5-xxl/")),
        wan_paths=wan_paths,
        wan_config_path=args.wan_config_path
    )


    ckpt = load_file(args.lora_path)

    lora_sd = {}
    adapter_sd = {}

    for k, v in ckpt.items():
        if ".lora_" in k:
            lora_sd[k] = v
        elif "i2v_adapter" in k:
            adapter_sd[k] = v

    pipe.load_lora(pipe.dit, state_dict=lora_sd, alpha=1)
    pipe.dit.load_state_dict(adapter_sd, strict=False)

    pipe.to("cuda")
    pipe.to(dtype=torch.bfloat16)


    dataset_path = args.dataset_path

    output_dir = os.path.join(dataset_path, "result")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = UnifiedDataset(
        base_path=dataset_path,
        data_file_keys=("src_video", "src_image"),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=dataset_path,
            height=704,
            width=1248,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=lambda x: x[0],
        num_workers=1
    )

    for batch_idx, batch in enumerate(dataloader):
        target_text = batch["prompt"]
        src_video = batch["src_video"]
        src_image = batch["src_image"]

        path = batch["path"]
        filename = os.path.basename(path).split(".")[0]


        video = pipe(
            prompt=target_text,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            src_video=src_video,
            input_image=src_image[0],
            height=704, width=1248,
            cfg_scale=5.0,
            num_frames=num_frames+4,
            num_inference_steps=50,
            seed=0, tiled=True
        )


        video_frames = list(video)

        last_frame = video_frames[-1]
        last_frame_np = np.array(last_frame)
        
        img_save_path = os.path.join(output_dir, f"{filename}.png")
        imageio.imwrite(img_save_path, last_frame_np)