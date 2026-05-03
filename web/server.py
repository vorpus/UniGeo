"""FastAPI server for the UniGeo web UI.

Endpoints:
  POST /api/upload      multipart image -> {session_id, meta}
  GET  /api/points/{id} binary point cloud (uint32 N || xyz f32 || rgb u8)
  GET  /api/meta/{id}   meta.json
  POST /api/generate    {session_id, x,y,z,phi,theta} -> {result_url}
  WS   /ws/{id}         streams progress events
  GET  /api/result/{id} the generated PNG
  GET  /                static UI
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent
WORKSPACE = WEB / "workspace"
STATIC = WEB / "static"
WORKSPACE.mkdir(exist_ok=True)

CKPTS = REPO / "checkpoints"
WAN_DIR = CKPTS / "Wan2.2-TI2V-5B"
LORA_PATH = CKPTS / "UniGeo" / "UniGeo_lora.safetensors"
WAN_CONFIG = REPO / "my_config.json"
VGGT_DIR = CKPTS / "VGGT-1B"

EXTRACT_PC = WEB / "extract_pc.py"
PYTHON = sys.executable

# ---------- progress channel ----------

class ProgressBus:
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def attach_loop(self, loop):
        self.loop = loop

    def get_queue(self, session_id: str) -> asyncio.Queue:
        q = self.queues.get(session_id)
        if q is None:
            q = asyncio.Queue()
            self.queues[session_id] = q
        return q

    def push(self, session_id: str, event: dict):
        # Safe to call from any thread.
        q = self.get_queue(session_id)
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(q.put_nowait, event)


bus = ProgressBus()
worker = None  # lazy-loaded WanWorker


# ---------- request models ----------

class GenerateReq(BaseModel):
    session_id: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    phi: float = 0.0  # yaw degrees
    theta: float = 0.0  # pitch degrees
    steps: int = 1     # number of intermediate end-positions to render


# ---------- helpers ----------

def session_dir(session_id: str) -> Path:
    return WORKSPACE / session_id


def load_points(session_id: str):
    import struct
    import numpy as np

    p = session_dir(session_id) / "points.bin"
    with open(p, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        xyz = np.frombuffer(f.read(n * 12), dtype=np.float32).reshape(n, 3)
        rgb = np.frombuffer(f.read(n * 3), dtype=np.uint8).reshape(n, 3)
    return xyz, rgb


# ---------- app ----------

app = FastAPI()


@app.on_event("startup")
async def _startup():
    bus.attach_loop(asyncio.get_running_loop())


@app.post("/api/upload")
async def upload(image: UploadFile = File(...)):
    session_id = uuid.uuid4().hex[:12]
    sd = session_dir(session_id)
    sd.mkdir(parents=True, exist_ok=True)

    # Save the upload, normalize to landscape PNG resized to model resolution.
    raw_path = sd / f"raw{Path(image.filename).suffix or '.png'}"
    with open(raw_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    img = Image.open(raw_path).convert("RGB")
    if img.height > img.width:
        raise HTTPException(400, "input image must be landscape (width >= height)")

    target_w, target_h = 1248, 704
    # Crop to 1248:704 ratio, then resize.
    src_ratio = img.width / img.height
    tgt_ratio = target_w / target_h
    if src_ratio > tgt_ratio:
        # too wide, crop sides
        new_w = int(img.height * tgt_ratio)
        x0 = (img.width - new_w) // 2
        img = img.crop((x0, 0, x0 + new_w, img.height))
    else:
        new_h = int(img.width / tgt_ratio)
        y0 = (img.height - new_h) // 2
        img = img.crop((0, y0, img.width, y0 + new_h))
    img = img.resize((target_w, target_h), Image.LANCZOS)
    input_png = sd / "input.png"
    img.save(input_png)

    # Run VGGT in a subprocess (frees its VRAM after).
    bus.push(session_id, {"stage": "vggt", "progress": 0.0, "msg": "extracting point cloud..."})

    proc = await asyncio.create_subprocess_exec(
        PYTHON, str(EXTRACT_PC),
        "--model_path", str(VGGT_DIR),
        "--image", str(input_png),
        "--out_dir", str(sd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        msg = out.decode(errors="replace")[-2000:]
        bus.push(session_id, {"stage": "error", "msg": msg})
        raise HTTPException(500, f"VGGT extraction failed:\n{msg}")

    bus.push(session_id, {"stage": "vggt", "progress": 1.0, "msg": "point cloud ready"})

    meta = json.loads((sd / "meta.json").read_text())
    return {"session_id": session_id, "meta": meta}


@app.get("/api/points/{session_id}")
def points(session_id: str):
    p = session_dir(session_id) / "points.bin"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/octet-stream")


@app.get("/api/meta/{session_id}")
def meta(session_id: str):
    p = session_dir(session_id) / "meta.json"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/json")


@app.post("/api/generate")
async def generate(req: GenerateReq):
    sd = session_dir(req.session_id)
    if not (sd / "input.png").exists():
        raise HTTPException(404, "session not found")

    global worker
    if worker is None:
        from web.pipeline import WanWorker
        bus.push(req.session_id, {"stage": "load", "progress": 0.0, "msg": "loading Wan2.2 (~60s, only on first generate)"})
        worker = WanWorker(str(WAN_DIR), str(LORA_PATH), str(WAN_CONFIG))
        await asyncio.to_thread(worker.load)
        bus.push(req.session_id, {"stage": "load", "progress": 1.0, "msg": "Wan2.2 ready"})

    from web.pipeline import render_trajectory_mp4

    xyz, rgb = load_points(req.session_id)
    meta = json.loads((sd / "meta.json").read_text())

    # Clear any prior step results so the gallery reflects this run.
    for old in sd.glob("result_*.png"):
        old.unlink()
    for old in sd.glob("result_*.mp4"):
        old.unlink()

    n_steps = max(1, min(req.steps, 20))
    t0 = time.time()
    result_urls = []
    video_url = None

    for k in range(1, n_steps + 1):
        a = k / n_steps  # fraction of the full delta for this step
        end = (req.x * a, req.y * a, req.z * a, req.phi * a, req.theta * a)
        traj_mp4 = sd / f"trajectory_{k}.mp4"
        out_png = sd / f"result_{k}.png"
        out_mp4 = sd / f"result_{k}.mp4"

        bus.push(req.session_id, {
            "stage": "render", "progress": 0.0,
            "step": k, "total_steps": n_steps,
            "msg": f"step {k}/{n_steps}: rendering camera trajectory",
        })

        def _render(end=end, traj_mp4=traj_mp4):
            render_trajectory_mp4(
                image_path=str(sd / "input.png"),
                points_xyz=xyz,
                points_rgb=rgb,
                image_size_hw=(meta["image_h"], meta["image_w"]),
                focals=meta["focals"],
                principal=meta["principal"],
                end_xyz_phi_theta=end,
                out_mp4=str(traj_mp4),
            )

        await asyncio.to_thread(_render)
        bus.push(req.session_id, {
            "stage": "render", "progress": 1.0,
            "step": k, "total_steps": n_steps,
            "msg": f"step {k}/{n_steps}: trajectory rendered",
        })

        bus.push(req.session_id, {
            "stage": "infer", "progress": 0.0,
            "step": k, "total_steps": n_steps,
            "msg": f"step {k}/{n_steps}: diffusion sampling (50 steps)",
        })

        def _cb(stage: str, progress: float, k=k, n_steps=n_steps):
            bus.push(req.session_id, {
                "stage": stage, "progress": progress,
                "step": k, "total_steps": n_steps,
            })

        def _infer(out_png=out_png, out_mp4=out_mp4, traj_mp4=traj_mp4):
            worker.infer(
                src_image_path=str(sd / "input.png"),
                src_video_path=str(traj_mp4),
                out_png_path=str(out_png),
                out_mp4_path=str(out_mp4),
                progress_cb=_cb,
            )

        await asyncio.to_thread(_infer)
        result_urls.append(f"/api/result/{req.session_id}/{k}.png")
        video_url = f"/api/result/{req.session_id}/{k}.mp4"

    bus.push(req.session_id, {
        "stage": "done", "progress": 1.0,
        "step": n_steps, "total_steps": n_steps,
        "msg": f"done in {time.time()-t0:.1f}s",
        "result_urls": result_urls,
        "video_url": video_url,
    })

    return {"result_urls": result_urls, "video_url": video_url}


@app.get("/api/result/{session_id}/{filename}")
def result_step(session_id: str, filename: str):
    # filename is like "1.png" or "2.mp4" -- map to result_<n>.<ext>.
    p = session_dir(session_id) / f"result_{filename}"
    if not p.exists():
        raise HTTPException(404)
    media = "image/png" if filename.endswith(".png") else "video/mp4"
    return FileResponse(p, media_type=media)


@app.get("/api/input/{session_id}")
def input_img(session_id: str):
    p = session_dir(session_id) / "input.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.websocket("/ws/{session_id}")
async def ws(ws: WebSocket, session_id: str):
    await ws.accept()
    q = bus.get_queue(session_id)
    try:
        while True:
            # Race the queue against the client closing or the server shutting down.
            recv_task = asyncio.create_task(ws.receive())
            get_task = asyncio.create_task(q.get())
            done, pending = await asyncio.wait({recv_task, get_task}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            if recv_task in done:
                msg = recv_task.result()
                if msg.get("type") == "websocket.disconnect":
                    return
            if get_task in done:
                ev = get_task.result()
                await ws.send_text(json.dumps(ev))
    except WebSocketDisconnect:
        pass


app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
