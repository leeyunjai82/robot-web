import asyncio
import signal
from contextlib import asynccontextmanager
import subprocess
import threading
import time
import os
import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Try importing ultralytics (YOLO)
try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    yolo_available = False
    print("WARNING: ultralytics not installed. YOLO segmentation disabled.")

# ─────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    global cap, yolo_model, music_files

    # servo 권한 설정
    setup_servo_permission()
    # 마이크 볼륨 80%
    setup_mic_volume(80)

    # Open camera
    cap = cv2.VideoCapture("/dev/video0")
    if not cap.isOpened():
        print("ERROR: Cannot open /dev/video0")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Camera opened: /dev/video0")

    # Load YOLO model
    if yolo_available:
        try:
            yolo_model = YOLO("yolo26n-seg.pt")
            print("YOLO model loaded: yolo11n-seg.pt")
        except Exception as e:
            print(f"YOLO load failed: {e}")

    # Scan music files
    music_files = scan_music()
    print(f"Found {len(music_files)} music files")

    yield  # ── 서버 실행 중 ──

    global running, recording_process, playback_process
    running = False
    time.sleep(0.2)
    if cap:
        cap.release()
    if recording_process:
        recording_process.terminate()
    if playback_process:
        playback_process.terminate()
    print("Server shutdown complete.")

app = FastAPI(title="Robot Control Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
running = True          # 스트리밍 루프 제어 플래그
camera_lock = threading.Lock()
cap: Optional[cv2.VideoCapture] = None

yolo_model = None
segmentation_enabled = False

# Motor state
pan_angle = 0
tilt_angle = 0

# Audio state
recording_process: Optional[subprocess.Popen] = None
playback_process: Optional[subprocess.Popen] = None
is_recording = False
RECORDING_FILE = "/tmp/robot_recording.wav"

# Music files search paths
MUSIC_DIRS = [
    os.path.expanduser("~/Music"),
    "/usr/share/sounds",
    "/home",
]
music_files: list = []


# ─────────────────────────────────────────────
# System setup helpers
# ─────────────────────────────────────────────
def setup_mic_volume(volume: int = 80):
    """DECXIN 마이크 볼륨 설정 (card 1). amixer로 Capture 볼륨 조정."""
    cmds = [
        # Capture 볼륨 (녹음 입력 게인)
        ["amixer", "-c", "1", "sset", "Mic", f"{volume}%", "cap"],
        ["amixer", "-c", "1", "sset", "Capture", f"{volume}%"],
        # 일부 USB 마이크는 컨트롤 이름이 다를 수 있어 두 번 시도
        ["amixer", "-c", "1", "set", "Mic Capture Volume", f"{volume}%"],
    ]
    success = False
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                print(f"Mic volume set to {volume}%: {' '.join(cmd)}")
                success = True
                break
        except Exception:
            continue
    if not success:
        print("WARNING: Could not set mic volume (control name may differ)")


def setup_servo_permission():
    """/dev/ttyACM0 를 ptl:ptl 소유로 변경해 sudo 없이 servo 명령 사용 가능하게."""
    tty = "/dev/ttyACM0"
    try:
        subprocess.run(["sudo", "chown", "ptl:ptl", tty], timeout=3, check=True)
        print(f"Permission set: chown ptl:ptl {tty}")
    except Exception as e:
        print(f"WARNING: chown {tty} failed: {e}")




def scan_music():
    found = []
    extensions = ["*.mp3", "*.wav", "*.ogg", "*.flac", "*.m4a"]
    for d in MUSIC_DIRS:
        if os.path.exists(d):
            for ext in extensions:
                found.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))
    return found


# ─────────────────────────────────────────────
# Frame generators
# ─────────────────────────────────────────────
def generate_raw_frames():
    """Raw camera stream."""
    global running
    while running:
        with camera_lock:
            if cap is None or not cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = cap.read()

        if not ret:
            time.sleep(0.05)
            continue

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


def generate_seg_frames():
    """YOLO segmentation stream."""
    global running
    while running:
        with camera_lock:
            if cap is None or not cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = cap.read()

        if not ret:
            time.sleep(0.05)
            continue

        if yolo_model is not None:
            try:
                results = yolo_model(frame, verbose=False)
                annotated = results[0].plot()
            except Exception:
                annotated = frame
        else:
            annotated = frame.copy()
            cv2.putText(annotated, "YOLO not available", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


# ─────────────────────────────────────────────
# Video routes
# ─────────────────────────────────────────────
@app.get("/stream/raw")
def stream_raw():
    return StreamingResponse(
        generate_raw_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/stream/seg")
def stream_seg():
    return StreamingResponse(
        generate_seg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/yolo/status")
def yolo_status():
    return {"available": yolo_available, "model_loaded": yolo_model is not None}


# ─────────────────────────────────────────────
# Pan / Tilt control
# ─────────────────────────────────────────────
class MotorCmd(BaseModel):
    pan: Optional[float] = None   # degrees -30 ~ 30
    tilt: Optional[float] = None  # degrees -10 ~ 10


def run_servo(motor_id: int, angle_deg: float):
    """Execute: servo write <id> <angle*10>  (chown으로 ptl:ptl 권한 설정됨)"""
    val = int(round(angle_deg * 10))
    cmd = ["./servo", "write", str(motor_id), str(val)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


@app.post("/api/motor")
def set_motor(cmd: MotorCmd):
    global pan_angle, tilt_angle
    results = {}

    if cmd.pan is not None:
        pan = max(-30, min(30, cmd.pan))
        ok, out, err = run_servo(0, pan)
        if ok:
            pan_angle = pan
        results["pan"] = {"angle": pan, "ok": ok, "err": err}

    if cmd.tilt is not None:
        tilt = max(-10, min(10, cmd.tilt))
        ok, out, err = run_servo(1, tilt)
        if ok:
            tilt_angle = tilt
        results["tilt"] = {"angle": tilt, "ok": ok, "err": err}

    return {"status": "ok", "results": results, "current": {"pan": pan_angle, "tilt": tilt_angle}}


@app.get("/api/motor")
def get_motor():
    return {"pan": pan_angle, "tilt": tilt_angle}


@app.post("/api/motor/reset")
def reset_motor():
    global pan_angle, tilt_angle
    run_servo(0, 0)
    run_servo(1, 0)
    pan_angle = 0
    tilt_angle = 0
    return {"status": "ok", "pan": 0, "tilt": 0}


# ─────────────────────────────────────────────
# Audio recording
# ─────────────────────────────────────────────
@app.post("/api/audio/record/start")
def start_recording():
    global recording_process, is_recording
    if is_recording:
        return {"status": "already_recording"}

    # DECXIN: card 1, device 0
    cmd = [
        "arecord",
        "-D", "hw:1,0",
        "-f", "cd",           # 16-bit, 44100Hz, stereo
        "-t", "wav",
        RECORDING_FILE
    ]
    try:
        recording_process = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        is_recording = True
        return {"status": "recording_started", "file": RECORDING_FILE}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/audio/record/stop")
def stop_recording():
    global recording_process, is_recording
    if not is_recording or recording_process is None:
        return {"status": "not_recording"}

    recording_process.terminate()
    recording_process.wait()
    recording_process = None
    is_recording = False

    exists = os.path.exists(RECORDING_FILE)
    size = os.path.getsize(RECORDING_FILE) if exists else 0
    return {"status": "stopped", "file": RECORDING_FILE, "size_bytes": size}


@app.post("/api/audio/record/play")
def play_recording():
    global playback_process
    if not os.path.exists(RECORDING_FILE):
        raise HTTPException(status_code=404, detail="No recording found")

    if playback_process and playback_process.poll() is None:
        playback_process.terminate()

    cmd = ["aplay", RECORDING_FILE]
    playback_process = subprocess.Popen(cmd)
    return {"status": "playing", "file": RECORDING_FILE}


@app.get("/api/audio/record/status")
def recording_status():
    return {
        "is_recording": is_recording,
        "file_exists": os.path.exists(RECORDING_FILE),
        "file_size": os.path.getsize(RECORDING_FILE) if os.path.exists(RECORDING_FILE) else 0,
    }


# ─────────────────────────────────────────────
# Music playback
# ─────────────────────────────────────────────
current_music_index = 0
music_loop = False


@app.get("/api/music/list")
def list_music():
    global music_files
    music_files = scan_music()
    return {
        "files": [{"index": i, "name": Path(f).name, "path": f} for i, f in enumerate(music_files)],
        "count": len(music_files),
    }


class MusicPlay(BaseModel):
    index: Optional[int] = None
    loop: bool = False


@app.post("/api/music/play")
def play_music(req: MusicPlay):
    global playback_process, current_music_index, music_loop, music_files

    if not music_files:
        music_files = scan_music()

    if not music_files:
        # Play system bell / test tone as fallback
        cmd = ["paplay", "/usr/share/sounds/alsa/Front_Center.wav"]
        try:
            playback_process = subprocess.Popen(cmd, stderr=subprocess.PIPE)
            return {"status": "playing_test_tone"}
        except Exception:
            raise HTTPException(status_code=404, detail="No music files found. Place mp3/wav files in ~/Music/")

    idx = req.index if req.index is not None else current_music_index
    idx = idx % len(music_files)
    current_music_index = idx
    music_loop = req.loop

    if playback_process and playback_process.poll() is None:
        playback_process.terminate()
        time.sleep(0.2)

    f = music_files[idx]
    # Try mpg123 for mp3, aplay for wav, ffplay for others
    ext = Path(f).suffix.lower()
    if ext == ".mp3":
        cmd = ["mpg123", "-q", f]
    elif ext == ".wav":
        cmd = ["aplay", f]
    else:
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", f]

    try:
        playback_process = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        return {"status": "playing", "file": Path(f).name, "index": idx, "loop": music_loop}
    except FileNotFoundError as e:
        # fallback to aplay
        try:
            playback_process = subprocess.Popen(["aplay", f], stderr=subprocess.PIPE)
            return {"status": "playing", "file": Path(f).name, "index": idx}
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Playback failed: {e2}")


@app.post("/api/music/stop")
def stop_music():
    global playback_process
    if playback_process and playback_process.poll() is None:
        playback_process.terminate()
        return {"status": "stopped"}
    return {"status": "not_playing"}


@app.post("/api/music/next")
def next_music():
    global current_music_index
    if not music_files:
        raise HTTPException(status_code=404, detail="No music files")
    current_music_index = (current_music_index + 1) % len(music_files)
    return play_music(MusicPlay(index=current_music_index, loop=music_loop))


@app.post("/api/music/prev")
def prev_music():
    global current_music_index
    if not music_files:
        raise HTTPException(status_code=404, detail="No music files")
    current_music_index = (current_music_index - 1) % len(music_files)
    return play_music(MusicPlay(index=current_music_index, loop=music_loop))


@app.get("/api/music/status")
def music_status():
    playing = playback_process is not None and playback_process.poll() is None
    return {
        "is_playing": playing,
        "current_index": current_music_index,
        "current_file": Path(music_files[current_music_index]).name if music_files else None,
        "total_files": len(music_files),
        "loop": music_loop,
    }


# ─────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn

    def handle_exit(sig, frame):
        global running
        print("\nShutting down gracefully...")
        running = False

        def force_quit():
            time.sleep(1.5)
            # 터미널 상태 복구 후 종료 (SSH 먹통 방지)
            try:
                subprocess.run(["stty", "sane"], timeout=1)
            except Exception:
                pass
            os._exit(0)

        threading.Thread(target=force_quit, daemon=True).start()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_graceful_shutdown=2,   # 최대 2초 대기 후 강제 종료
    )
