from .operators import *
import torch, json

def save_video_tensor_as_mp4(video_frames, out_path, fps=8):


    # (C,T,H,W) -> (T,H,W,C)
    video_np = []
    for frame in video_frames:
        
        frame_np = np.array(frame)
        video_np.append(frame_np)
    
    
    video = np.stack(video_np, axis=0)

    imageio.mimwrite(
        out_path,
        video,
        fps=fps,
        codec="libx264",
        quality=8,
    )


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
    ):
        self.base_path = base_path
        self.repeat = repeat
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
        src_dir = os.path.join(self.base_path, "point_video")
        tgt_dir = os.path.join(self.base_path, "videos/train")

        video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")

        for fname in os.listdir(src_dir):
            if not fname.lower().endswith(video_exts):
                continue

            src_path = os.path.join(src_dir, fname)
            tgt_path = os.path.join(tgt_dir, fname)

            if not os.path.exists(tgt_path) or os.path.getsize(tgt_path) == 0:
                print(f"跳过无效文件：{tgt_path}")
                continue
            if not os.path.exists(src_path) or os.path.getsize(src_path) == 0:
                print(f"跳过无效文件：{src_path}")
                continue

            self.data.append({
                "src_video": src_path,
                "tgt_video": tgt_path,
                "prompt": "Ensure the consistency of the video"
            })

        print(f"Found {len(self.data)} video pairs")



    def __getitem__(self, data_id):

        try:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    data[key] = self.main_data_operator(data[key])
            return data
        except Exception:
            return self.__getitem__(data_id + 1)

    def __len__(self):
        return len(self.data) * self.repeat
