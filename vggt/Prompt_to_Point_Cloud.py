import argparse
import torch, imageio, os, json, re
from scipy.spatial.transform import Rotation
import cv2
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from PIL import Image
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
from utils import to_numpy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    return parser.parse_args()


def setup_renderer(cameras, image_size):
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius = 0.01,
        points_per_pixel = 10,
        bin_size = 0
    )

    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )

    render_setup =  {'cameras': cameras, 'raster_settings': raster_settings, 'renderer': renderer}

    return render_setup


def render_pcd(pts3d, imgs, masks, views, renderer, device, nbv=False):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)

    if masks is None:
        pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
        col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
    else:
        pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
        col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)
    
    point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
    images = renderer(point_cloud)

    if nbv:
        color_mask = torch.ones(col.shape).to(device)
        point_cloud_mask = Pointclouds(points=[pts], features=[color_mask]).extend(views)
        view_masks = renderer(point_cloud_mask)
    else: 
        view_masks = None

    return images, view_masks


def run_render(pcd, imgs, masks, H, W, camera_traj, num_views, device, nbv=True):
    render_setup = setup_renderer(camera_traj, image_size=(H,W))
    renderer = render_setup['renderer']
    render_results, viewmask = render_pcd(pcd, imgs, masks, num_views, renderer, device, nbv=nbv)
    return render_results, viewmask


def build_estimate_rel(x, y, z, phi, theta):

    delta_euler = [theta, phi, 0.0]

    rot_mat = Rotation.from_euler('xyz', delta_euler, degrees=True).as_matrix()

    estimate_rel = np.eye(4)
    estimate_rel[:3, :3] = rot_mat
    estimate_rel[:3, 3] = [x, y, z]

    return estimate_rel


def parse_prompt_to_motion(prompt):
    prompt = prompt.lower()
    x = y = z = phi = theta = 0.0

    clauses = re.split(r'[;,\n]| and ', prompt)

    for clause in clauses:
        
        nums = re.findall(r"[-+]?\d*\.?\d+", clause)
        
        if not nums:
            continue
            
        val = float(nums[0])

        if "pans left" in clause:
            phi = -val   
        elif "pans right" in clause:
            phi = val
        elif "tilts up" in clause:
            theta = val
        elif "tilts down" in clause:
            theta = -val
        elif "moves forward" in clause:
            z = val  
        elif "moves backward" in clause:
            z = -val
        elif "moves up" in clause:
            y = -val
        elif "moves down" in clause:
            y = val
        elif "moves left" in clause:
            x = -val
        elif "moves right" in clause:
            x = val

    print(f"Parsed motion from prompt: x={x}, y={y}, z={z}, phi={phi}, theta={theta}")
    return x, y, z, phi, theta


def generate_all_motions_from_prompt(prompt, num_frames):

    x, y, z, phi, theta = parse_prompt_to_motion(prompt)

    results = []

    for i in range(num_frames):
        alpha = i / (num_frames - 1)

        results.append((
            x * alpha,
            y * alpha,
            z * alpha,
            phi * alpha,
            theta * alpha
        ))

    return results


def save_image_list_as_mp4(image_list, out_path, fps=30):

    imageio.mimwrite(
        out_path,
        image_list,
        fps=fps,
        codec="libx264",  
        quality=8,       
    )
    print(f"video saved to: {out_path}")


args = get_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


num_frames = 81

model = VGGT.from_pretrained(args.model_path).to(device)

dataset_path = args.dataset_path
imgs_path = [img_path for img_path in os.listdir(dataset_path) if img_path.endswith(".png")]
paths = [os.path.join(dataset_path, img_path) for img_path in imgs_path]
camera_prompt = os.path.join(dataset_path, "prompt.json")
with open(camera_prompt, 'r') as f:
    camera_prompt_data = json.load(f) 
result_folder = os.path.join(dataset_path, "point_cloud")
os.makedirs(result_folder, exist_ok=True)


for path in paths:

    img = Image.open(path).convert("RGB")

    prompt = camera_prompt_data[os.path.basename(path)]

    all_steps = generate_all_motions_from_prompt(prompt, num_frames=num_frames)

    cam_idx = list(range(num_frames))
    traj = [build_estimate_rel(*all_steps[idx]) for idx in cam_idx]

    first_frame = [img, img]
    first_frame = load_and_preprocess_images(first_frame)
    first_frame = first_frame.to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(first_frame)
            depth_map = predictions["depth"]

            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], first_frame.shape[-2:])

            first_frame_world_points = predictions["world_points"][0][0]

            focals = intrinsic[0][0][:2, :2].diag().unsqueeze(0).to(device)
            principal_points = intrinsic[0][0][:2, 2].unsqueeze(0).to(device)

    raw_image = first_frame[0].cpu().numpy() 
    raw_image = raw_image.transpose(1, 2, 0)

    render_results_list = []

    for estimate_rel in traj:
        estimate_rel = torch.from_numpy(estimate_rel).float().to(device)
        relative_c2ws = estimate_rel.unsqueeze(0)
        R, T = relative_c2ws[:, :3, :3], relative_c2ws[:, :3, 3:]
        R = torch.stack([-R[:, :, 0], -R[:, :, 1], R[:, :, 2]], 2)
        new_c2w = torch.cat([R, T], 2)
    
        w2c = torch.linalg.inv(torch.cat(
            (new_c2w, torch.Tensor([[[0, 0, 0, 1]]]).to(device).repeat(new_c2w.shape[0], 1, 1)), 
            1
        ))
        R_new, T_new = w2c[:, :3, :3].permute(0, 2, 1), w2c[:, :3, 3]


        image_size = (first_frame.shape[-2:],)

        cameras = PerspectiveCameras(
            focal_length=focals, 
            principal_point=principal_points, 
            in_ndc=False, 
            image_size=image_size, 
            R=R_new, 
            T=T_new, 
            device=device
        )

        masks = None
        render_results, viewmask = run_render(
            [first_frame_world_points], 
            [raw_image], 
            masks, 
            image_size[0][0], image_size[0][1], 
            cameras, 
            1, 
            device=device
        )

        
        render_result = (render_results[-1].detach().cpu().numpy() * 255).astype(np.uint8)
    
        if len(render_result.shape) == 2:
            render_result = cv2.cvtColor(render_result, cv2.COLOR_GRAY2RGB)
        elif render_result.shape[-1] == 4:
            render_result = render_result[..., :3]

        render_results_list.append(render_result)

    raw_image = first_frame[0].cpu().numpy() 
    raw_image = raw_image.transpose(1, 2, 0)

    raw_image = (raw_image * 255).clip(0, 255).astype(np.uint8)

    render_results_list[0] = raw_image

    save_image_list_as_mp4(render_results_list, os.path.join(result_folder, f"{os.path.basename(path).split('.')[0]}.mp4"), fps=30)