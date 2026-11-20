"""ゴルフスイング分析 — 最初のStreamlit版を基にした簡単操作版。
起動: python -m streamlit run app.py
"""
from __future__ import annotations
import base64
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
import uuid
import zipfile

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import streamlit as st
from streamlit.components.v1 import declare_component

P_CODES = [f"P{i}" for i in range(1, 11)]
P_NAMES = dict(zip(P_CODES, ["アドレス", "テイクバック", "ハーフウェイバック", "トップ", "リード腕が水平（ダウンスイング）", "デリバリー", "インパクト", "フォロー", "ハイフォロー", "フィニッシュ"]))
LINE_NAMES = {"head":"頭の円（追従＋P1基準）", "spine":"胸と腰の中心線（前傾角）", "tilt":"胸と腰の傾き（正面）", "shoulders":"両肩のライン", "shaft":"シャフトプレーン", "hip":"ヒップライン", "ankles":"両足首の垂線", "vertical":"腰中心の垂直基準線", "back_vertical":"垂直基準線（身長程度）", "hip_right_knee":"腰の中心〜右ひざの中心"}
COLORS = {"head":"#43D9E6", "spine":"#FFFF00", "tilt":"#FFA500", "shoulders":"#0FDE32", "shaft":"#FFFFFF", "hip":"#FFCC55", "ankles":"#FFFF00", "vertical":"#43D9E6", "back_vertical":"#FFFFFF", "hip_right_knee":"#FFFF00"}
LINE_NAMES["right_leg_outer"] = "右脚の外側（P1基準線）"
COLORS["right_leg_outer"] = "#1BC4D4"
REAR_ONLY_LINES = ("back_vertical", "hip_right_knee")
FRONT_ONLY_LINES = ("right_leg_outer",)
DEFAULT_LINES = ["head", "spine", "shoulders", "shaft", "hip", *REAR_ONLY_LINES]
FRONT_DEFAULT_LINES = ["head", "tilt", "shoulders", "ankles", *FRONT_ONLY_LINES]
RIGHT_LEG_POINTS = ("right_ankle_outer", "right_thigh_outer")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"


def run_ffmpeg(args: list[str], timeout: int = 240) -> None:
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("動画の変換に失敗しました。\n" + result.stderr.decode("utf-8", errors="replace")[-1800:])


def video_info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("動画を開けませんでした。端末内に保存したMP4動画を選び直してください。")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        if not ok or w <= 0 or h <= 0 or not math.isfinite(fps) or fps <= 0 or n < 2:
            raise ValueError("動画の映像や長さを取得できませんでした。別のMP4動画をお試しください。")
        return {"width": w, "height": h, "fps": fps, "frames": n, "seconds": n/fps}
    finally:
        cap.release()


def normalize_video(src: Path, dest: Path, max_side: int = 960) -> dict:
    # 全コマをRAMへ保持せず、向き・サイズ・フレームレートを先に統一する。
    scale = f"scale=w='min({max_side},iw)':h='min({max_side},ih)':force_original_aspect_ratio=decrease:force_divisible_by=2"
    run_ffmpeg(["-i", src, "-map", "0:v:0", "-vf", scale + ",fps='min(source_fps,60)'", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2", dest])
    return video_info(dest)


def ensure_model() -> Path:
    folder = Path(os.environ.get("GOLF_MODEL_DIR", str(Path.home()/".cache"/"golf_pose_simple")))
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "pose_landmarker_heavy.task"
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    tmp = folder / (uuid.uuid4().hex + ".part")
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent":"GolfSwingSimple/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        if tmp.stat().st_size < 1_000_000:
            raise ValueError("AIモデルのダウンロードが完了しませんでした。")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def pose_to_dict(lm, width: int, height: int) -> dict:
    point = lambda i: [float(lm[i].x), float(lm[i].y)]
    mid = lambda i,j: [(lm[i].x+lm[j].x)/2, (lm[i].y+lm[j].y)/2]
    # 鼻だけでなく両耳も使い、正面・後方で頭の中心が極端にずれないようにする。
    head = [(lm[0].x+lm[7].x+lm[8].x)/3, (lm[0].y+lm[7].y+lm[8].y)/3]
    ear_width = math.hypot(lm[7].x-lm[8].x, (lm[7].y-lm[8].y)*height/width)
    left_hip, right_hip = point(23), point(24)
    left_ankle, right_ankle = point(27), point(28)
    pose = {"head":head, "radius":max(.018, min(.10, ear_width*.75)), "ls":point(11), "rs":point(12), "sh":mid(11,12), "hip":mid(23,24), "rk":point(26), "wr":mid(15,16), "la":left_ankle, "ra":right_ankle}
    wrist_visibilities = [float(lm[i].visibility) for i in (15, 16)]
    pose.update(lw=point(15), rw=point(16), wrist_visibility=max(wrist_visibilities),
                larm_visibility=min(float(lm[i].visibility) for i in (11, 15)),
                rarm_visibility=min(float(lm[i].visibility) for i in (12, 16)))
    if max(wrist_visibilities) >= .35:
        # Give the more visible wrist extra weight when the other hand is occluded.
        left_weight, right_weight = (max(.20, v) for v in wrist_visibilities)
        pose["tracked_wr"] = [
            (left * left_weight + right * right_weight) / (left_weight + right_weight)
            for left, right in zip(pose["lw"], pose["rw"])
        ]
    if all(valid_point(p) for p in (left_hip, right_hip, right_ankle)):
        # Pose landmarks are joint centers. Offset the endpoints toward the
        # outer edge in screen space to give the user an adjustable estimate.
        hip_delta = right_hip[0] - left_hip[0]
        leg_scale = abs(right_ankle[1] - right_hip[1]) * height / width
        side = hip_delta if abs(hip_delta) >= .04 * leg_scale else right_ankle[0] - left_ankle[0]
        outward = 1 if side > 0 else -1
        span = max(abs(hip_delta), .35 * leg_scale)
        for key, point_xy, factor in zip(RIGHT_LEG_POINTS, (right_ankle, right_hip), (.08, .22)):
            x, y = point_xy
            pose[key] = [min(1., max(0., x + outward * span * factor)), y]
    return pose


def analyze_skeleton(path: Path, info: dict, progress=None) -> list:
    # Python側のCPU解析。スマホのWebGL対応には依存しない。
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    options = vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model())), running_mode=vision.RunningMode.VIDEO, num_poses=1, min_pose_detection_confidence=.45, min_tracking_confidence=.45)
    cap = cv2.VideoCapture(str(path))
    poses = []
    try:
        with vision.PoseLandmarker.create_from_options(options) as detector:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(i*1000/info["fps"]))
                poses.append(pose_to_dict(result.pose_landmarks[0], info["width"], info["height"]) if result.pose_landmarks else None)
                i += 1
                if progress and (i % 5 == 0 or i == info["frames"]):
                    progress(min(1., i/max(1,info["frames"])), f"骨格解析 {i} / {info['frames']} コマ")
        if not poses:
            raise ValueError("解析対象の映像を読み取れませんでした。")
        return poses
    finally:
        cap.release()


def swing_pose_track(poses: list, fps: float, n: int, aspect: float, left_handed: bool):
    """Build a smoothed, torso-scaled track. Bridge only short tracking gaps."""
    lead, trail = ("r", "l") if left_handed else ("l", "r")
    points = np.full((n, 5, 2), np.nan)
    for i, pose in enumerate(poses[:n]):
        if not pose or pose.get("wrist_visibility", 1.) < .35:
            continue
        values = [
            pose.get("tracked_wr", pose.get("wr")), pose.get("hip"), pose.get("sh"),
            pose.get(lead + "s"), pose.get(trail + "s"),
        ]
        if not all(valid_point(point) for point in values[:3]):
            continue
        points[i, :3] = values[:3]
        for column, side in ((3, lead), (4, trail)):
            shoulder, wrist = values[column], pose.get(side + "w")
            if not valid_point(shoulder):
                continue
            if side + "w" not in pose:
                wrist = values[0]  # Workspace data saved before separate wrists were tracked.
            if valid_point(wrist) and pose.get(side + "arm_visibility", 1.) >= .40:
                points[i, column] = np.subtract(wrist, shoulder)
    points[:, :, 0] *= aspect
    torso = np.linalg.norm(points[:, 1] - points[:, 2], axis=1)
    usable = torso[np.isfinite(torso) & (torso > .03)]
    if not len(usable):
        return None
    scale = float(np.median(usable))
    observed = np.isfinite(points[:, :, 0])
    smooth_radius = max(1, round(fps * .025))
    max_gap = max(1, round(fps * .10))
    weights = np.arange(1, smooth_radius + 2)
    kernel = np.r_[weights, weights[-2::-1]]
    kernel = kernel / kernel.sum()
    for column in range(points.shape[1]):
        valid = np.flatnonzero(observed[:, column])
        for block in np.split(valid, np.flatnonzero(np.diff(valid) > max_gap + 1) + 1):
            if not len(block):
                continue
            span = np.arange(block[0], block[-1] + 1)
            for axis in range(2):
                filled = np.interp(span, block, points[block, column, axis])
                padded = np.pad(filled, smooth_radius, mode="edge")
                points[span, column, axis] = np.convolve(padded, kernel, mode="valid")
    return points / scale, observed[:, 0]


def find_swing_anchors(wrist, height, speed, fps):
    """Look for a rise, a downswing and a follow-through in one tracked segment."""
    n = len(height)
    candidates = np.flatnonzero(
        (height[1:-1] >= height[:-2]) & (height[1:-1] > height[2:]) & (height[1:-1] > .35)
    ) + 1
    best = None
    for top in candidates:
        if top < 3 or n - top < 7:
            continue
        before = np.arange(max(0, top - round(6 * fps)), top - 2)
        low = before[height[before] < .20]
        if not len(low):
            continue
        # Use the last low-hand segment leading to this top, excluding earlier waggles.
        low = np.split(low, np.flatnonzero(np.diff(low) > 1) + 1)[-1]
        baseline = float(np.percentile(height[low], 30))
        settled = low[(height[low] <= baseline + .13) & (speed[low] <= .85)]
        setup = int(settled[-1] if len(settled) else low[0])
        rise = height[top] - height[setup]
        if rise < .58 or top - setup < max(3, round(.12 * fps)):
            continue
        downswing_limit = min(round(3 * fps), max(round(.65 * fps), top - setup))
        after = np.arange(top + 3, min(n - 3, top + downswing_limit))
        if not len(after):
            continue
        # A bottom near address height followed by a clear rise is consistent with impact.
        bottoms = after[
            (height[after] < height[after - 1])
            & (height[after] <= height[after + 1])
            & (height[after] < .25)
            & (height[top] - height[after] >= .55)
        ]
        for strike in bottoms:
            follow_span = min(n, strike + max(5, min(round(2.5 * fps), (strike - top) * 3)))
            follow_rise = np.max(height[strike + 1:follow_span]) - height[strike]
            if follow_rise < .45:
                continue
            attack = float(np.percentile(speed[top:strike + 1], 80))
            release = float(np.percentile(speed[strike:follow_span], 80))
            # The same segment should show a fast return and release near setup.
            delta = np.linalg.norm(wrist[strike] - wrist[setup])
            score = (
                min(rise, 2.5) + min(follow_rise, 2.5)
                + .40 * math.log1p(attack) + .30 * math.log1p(release)
                - .8 * delta - .30 * (strike - top) / fps
            )
            if best is None or score > best[0]:
                best = (score, setup, int(top), int(strike))
    return best


def auto_positions(poses: list, info: dict, left_handed: bool = False) -> tuple[dict, str]:
    n, fps = info["frames"], info["fps"]
    fallback = dict(zip(P_CODES, [round((n - 1) * i / 9) for i in range(10)]))
    track = swing_pose_track(poses, fps, n, info["width"] / info["height"], left_handed)
    if track is None:
        return fallback, "manual"
    points, observed = track
    visible = np.flatnonzero(np.isfinite(points[:, 0, 0]))
    swing = None
    for segment in np.split(visible, np.flatnonzero(np.diff(visible) > 1) + 1):
        if len(segment) < 10:
            continue
        first, last = int(segment[0]), int(segment[-1]) + 1
        wrist, hip = points[first:last, :2].transpose(1, 0, 2)
        height = hip[:, 1] - wrist[:, 1]
        speed = np.linalg.norm(np.gradient(wrist - hip, axis=0), axis=1) * fps
        anchors = find_swing_anchors(wrist, height, speed, fps)
        if anchors and (swing is None or anchors[0] > swing[0][0]):
            swing = anchors, first, height, speed
    if swing is None:
        return fallback, "manual"

    (_, p1, p4, p7), first, height, speed = swing
    arms = points[first:first + len(height), 3:, :]
    # The vertical component comes from the shoulder-to-wrist line in the image.
    arm_level = arms[:, :, 1] / np.maximum(.25, np.linalg.norm(arms, axis=2))
    follow_limit = min(round(5 * fps), max(round(1.1 * fps), 7 * (p7 - p4)))
    post_end = min(len(height), p7 + max(8, follow_limit))
    follow = np.arange(p7 + 1, post_end)
    crest = int(follow[np.argmax(height[follow])])
    if crest < p7 + 3:
        return fallback, "manual"

    # Finish is the first sustained low-motion segment after the hands rise.
    hold = max(2, round(.15 * fps))
    finish = None
    for index in range(max(p7 + 3, crest - hold + 1), post_end - hold + 1):
        window = slice(index, index + hold)
        if np.max(speed[window]) < .8 and np.min(height[window]) >= height[crest] - .30:
            finish = index
            break
    needs_review = finish is None
    p10 = finish if finish is not None else max(p7 + 3, crest)

    def pick(lo, hi, error, expected, max_error=None):
        nonlocal needs_review
        frames = np.arange(lo, hi + 1)
        valid = np.isfinite(error[frames])
        if not np.any(valid):
            needs_review = True
            return int(frames[np.argmin(abs(frames - expected))])
        frames = frames[valid]
        # In a flat/missing signal, favor the frame closest to the expected crossing.
        cost = abs(error[frames]) + .12 * abs(frames - expected) / max(1, hi - lo)
        cost += .10 * ~observed[frames + first]
        chosen = int(frames[np.argmin(cost)])
        if max_error is not None and abs(error[chosen]) > max_error:
            needs_review = True
        return chosen

    p3 = pick(p1 + 2, p4 - 1, arm_level[:, 0], p1 + .70 * (p4 - p1), .22)
    p5 = pick(p4 + 1, p7 - 2, arm_level[:, 0], p4 + .35 * (p7 - p4), .22)
    # Shaft-horizontal positions use hand height around the pelvis as a proxy.
    p2 = pick(p1 + 1, p3 - 1, height, (p1 + p3) / 2)
    p6 = pick(p5 + 1, p7 - 1, height, (p5 + p7) / 2)
    p9 = pick(p7 + 2, min(p10 - 1, max(p7 + 2, crest)), arm_level[:, 1], p7 + .62 * (p10 - p7), .22)
    p8 = pick(p7 + 1, p9 - 1, height, (p7 + p9) / 2)
    frames = (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)
    needs_review |= not all(observed[first + frame] for frame in frames)
    return dict(zip(P_CODES, (first + frame for frame in frames))), "review" if needs_review else "estimated"


def read_frame(path: str | Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0,int(index)))
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"{index}コマ目を読み取れませんでした。")
        return frame
    finally:
        cap.release()


def get_pose(job: dict, frame: int, phase: str | None = None) -> dict | None:
    if phase and phase in job["body_overrides"] and frame == job["p_frames"][phase]:
        return job["body_overrides"][phase]
    for code, body in job["body_overrides"].items():
        if frame == job["p_frames"][code]:
            return body
    return job["poses"][frame] if 0 <= frame < len(job["poses"]) else None


def initial_guides(ref: dict | None, info: dict) -> dict:
    # 検出できない場合もシャフト・ヒップの手動調整用の2本は操作可能にする。
    g = {"shaft":[[.70,.85],[.52,.52]], "hip":.35, "ankles":[.38,.62], "vertical":.5, "back_vertical":rear_reference_x(ref), "head":None, "radius":.04, "top":0., "bottom":1.}
    if not ref:
        return g
    front = 1 if ref["wr"][0] >= ref["hip"][0] else -1
    ground = min(.99,max(ref["la"][1],ref["ra"][1])+.025)
    ankle_mid = (ref["la"][0]+ref["ra"][0])/2
    g.update(shaft=[[float(np.clip(ankle_mid+front*.16,.01,.99)),ground],ref["wr"][:]], hip=float(np.clip(ref["hip"][0]-front*.06,.01,.99)), ankles=[float(np.clip(min(ref["la"][0],ref["ra"][0])-.015,.01,.99)),float(np.clip(max(ref["la"][0],ref["ra"][0])+.015,.01,.99))], vertical=ref["hip"][0], head=ref["head"][:],radius=ref["radius"],top=max(0.,ref["head"][1]-ref["radius"]*info["width"]/info["height"]*1.3),bottom=ground)
    return g


def rear_reference_x(ref: dict | None) -> float:
    return min(.99, max(.01, ref["ra"][0])) if ref else .65


def build_job(path: Path, info: dict, poses: list, folder: Path, name: str, left_handed: bool = False) -> dict:
    p_frames, position_status = auto_positions(poses, info, left_handed)
    ref = poses[p_frames["P1"]] if p_frames["P1"] < len(poses) else None
    ref = ref or next((p for p in poses if p),None)
    return {"token":uuid.uuid4().hex,"path":str(path),"folder":str(folder),"name":name,"info":info,"poses":poses,"p_frames":p_frames,"position_status":position_status,"guides":initial_guides(ref,info),"body_overrides":{},"guides_manual":False,"shapes":[],"last_event":None,"output_video":None,"output_images":None}


def in_positions(frame: int, selected: list, p_frames: dict) -> bool:
    return not selected or any(p_frames.get(code) == frame for code in selected)


def right_leg_reference(job: dict) -> tuple[list, list] | None:
    index = job["p_frames"]["P1"]
    ref = job["body_overrides"].get("P1")
    if not ref and 0 <= index < len(job["poses"]):
        ref = job["poses"][index]
    if ref and all(valid_point(ref.get(key)) for key in RIGHT_LEG_POINTS):
        ankle, thigh = RIGHT_LEG_POINTS
        return ref[ankle], ref[thigh]
    return None


def primitives_for(job: dict, frame: int, cfg: dict, phase: str | None = None, custom_positions: list | None = None) -> list:
    body, g = get_pose(job,frame,phase),job["guides"]
    out = []
    if job.get("show_p1_spine"):
        ref=get_pose(job,job["p_frames"]["P1"],"P1")
        if ref:
            out.append({"type":"line","id":"p1_spine_reference","a":ref["hip"][:],"b":ref["sh"][:],"angle":False,"color":"#FFFFFF","width":2,"dash":"dashed"})
    def add_line(code,a,b,angle=False,vertical=False):
        out.append({"type":"line","id":code,"a":a,"b":b,"angle":angle,"vertical":vertical,**cfg[code]})
    front = job.get("camera_mode") == "front"
    hidden_lines = REAR_ONLY_LINES if front else FRONT_ONLY_LINES
    for code,s in cfg.items():
        if not s["enabled"] or (code != "right_leg_outer" and not in_positions(frame,s["positions"],job["p_frames"])):
            continue
        if code in hidden_lines:
            continue
        if code == "head" and g.get("head"):
            out.append({**s,"type":"circle","id":"head_reference","center":g["head"][:],"radius":g["radius"],"dash":"dashed"})
            if body and body.get("head"):
                ref=get_pose(job,job["p_frames"]["P1"],"P1")
                origin=ref["head"] if ref and ref.get("head") else g["head"]
                delta=[body["head"][i]-origin[i] for i in range(2)]
                center=[g["head"][i]+delta[i] for i in range(2)]
                out.append({**s,"type":"circle","id":code,"center":center,"radius":g["radius"],"head_delta":delta})
        elif code in ("spine","tilt") and body:
            if code=="tilt":
                out.append({"type":"line","id":"tilt_zero","a":body["hip"][:],"b":[body["hip"][0],body["sh"][1]],"angle":False,"color":"#FFFFFF","width":2,"dash":"dashed"})
            add_line(code,body["hip"],body["sh"],True,code=="tilt")
            if code=="tilt":out[-1]["target_dir"]=job.get("target_dir",-1)
        elif code == "shoulders" and body:
            add_line(code,body["ls"],body["rs"],True)
            right=max((body["ls"],body["rs"]),key=lambda p:p[0])
            out[-1]["label_pos"]=[right[0]+.025,right[1]-.01]
        elif code == "shaft":
            a,b = g["shaft"]
            add_line(code,a,[a[0]+(b[0]-a[0])*2.2,a[1]+(b[1]-a[1])*2.2],True)
            out[-1]["label_pos"]=[a[0]+(b[0]-a[0])*.45+.014,a[1]+(b[1]-a[1])*.45+.015]
        elif code == "hip":
            add_line(code,[g["hip"],g["top"]],[g["hip"],g["bottom"]])
        elif code == "ankles":
            for k,x in enumerate(g["ankles"]):
                add_line(code,[x,g["top"]],[x,g["bottom"]])
                out[-1]["part"] = k
        elif code == "vertical":
            add_line(code,[g["vertical"],g["top"]],[g["vertical"],g["bottom"]])
        elif code == "back_vertical":
            x = g.get("back_vertical", .65)
            top, bottom = (g["top"], g["bottom"]) if g.get("head") else (.12, .90)
            add_line(code, [x, top], [x, bottom])
        elif code == "hip_right_knee" and body and valid_point(body.get("rk")):
            add_line(code, body["hip"], body["rk"], angle=True)
        elif code == "right_leg_outer":
            frames = job["p_frames"]
            if frames["P1"] <= frame <= frames["P4"] and (phase is None or phase in P_CODES[:4]):
                reference = right_leg_reference(job)
                if reference:
                    add_line(code, *reference, angle=True)
    for shape in job["shapes"]:
        target = shape.get("target_p")
        if (target and job["p_frames"].get(target) == frame) or (not target and in_positions(frame,custom_positions or [],job["p_frames"])):
            out.append(copy.deepcopy(shape))
    return out


def pil_font(size: int):
    for path in ["C:/Windows/Fonts/YuGothB.ttc","C:/Windows/Fonts/meiryo.ttc","/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc","/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path,size)
    return ImageFont.load_default(size=size)


def sportsbox_overlay(job: dict, phase: str) -> str | None:
    values = job.get("sportsbox_angles", {}).get(phase, {})
    if any(values.get(metric) is None for metric in SPORTSBOX_METRICS):
        return None
    angles = "   ".join(f"{name} {values[metric]}°" for metric, name in SPORTSBOX_METRICS.items())
    return f"回転角度   {angles}"


def draw_annotations(frame: np.ndarray, shapes: list, label: str | None = None, sportsbox: str | None = None) -> np.ndarray:
    im = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)
    w,h = im.size
    font = pil_font(max(14,round(26*h/1080)))
    def xy(p): return (round(p[0]*w),round(p[1]*h))
    def styled_line(a,b,color,width,dashed):
        a,b = xy(a),xy(b)
        if not dashed:
            draw.line([a,b],fill=color,width=width)
            return
        length = math.dist(a,b)
        dash = max(2,round(4*h/1080))
        for start in range(0,max(1,math.ceil(length)),dash*2):
            if not length: break
            t0,t1 = start/length,min(1.,(start+dash)/length)
            draw.line([(a[0]+(b[0]-a[0])*t0,a[1]+(b[1]-a[1])*t0),(a[0]+(b[0]-a[0])*t1,a[1]+(b[1]-a[1])*t1)],fill=color,width=width)
    for s in shapes:
        color = s.get("color","#FFFF00")
        width = max(1,round(s.get("width",3)*h/1080))
        dashed = s.get("dash") == "dashed"
        if s["type"] == "line":
            styled_line(s["a"],s["b"],color,width,dashed)
            if s.get("angle"):
                a,b = xy(s["a"]),xy(s["b"])
                angle = round(math.degrees(math.atan2(abs(b[1]-a[1]),abs(b[0]-a[0]))))
                if s.get("vertical"):
                    angle=round(math.degrees(math.atan2(abs(b[0]-a[0]),abs(b[1]-a[1]))))
                    if (b[0]-a[0])*s.get("target_dir",-1)<0:angle=-angle
                label_xy=xy(s["label_pos"]) if s.get("label_pos") else ((a[0]+b[0])/2+8,(a[1]+b[1])/2-12)
                pos=(max(2,min(w-65,label_xy[0])),max(2,min(h-30,label_xy[1])))
                draw.text(pos,f"{angle}°",font=font,fill=color,stroke_width=2,stroke_fill="#111111")
        elif s["type"] == "circle":
            cx,cy=xy(s["center"]);r=s["radius"]*w
            box=[cx-r,cy-r,cx+r,cy+r]
            if dashed:
                dash=max(2,round(4*h/1080))
                count=max(12,math.ceil(2*math.pi*r/(dash*2)))
                step=360/count
                for i in range(count):draw.arc(box,i*step,(i+.5)*step,fill=color,width=width)
            else: draw.ellipse(box,outline=color,width=width)
        elif s["type"] == "free":
            pts=s.get("points",[])
            for a,b in zip(pts,pts[1:]): styled_line(a,b,color,width,dashed)
    if label:
        draw.text((15,12),label,font=pil_font(max(22,round(36*h/1080))),fill="#FFE26B",stroke_width=2,stroke_fill="black")
    if sportsbox:
        size = max(8, min(max(15, round(h * .028)), w // 17))
        font = pil_font(size)
        stroke = max(1, round(size * .1))
        margin = max(6, round(size * .65))
        pad = max(4, round(size * .35))
        left, top, right, bottom = draw.textbbox(
            (0, 0), sportsbox, font=font, anchor="rt", stroke_width=stroke,
        )
        x = w - margin - pad - right
        y = margin + pad - top
        ImageDraw.Draw(im, "RGBA").rectangle(
            (x + left - pad, margin, w - margin, y + bottom + pad),
            fill=(0, 0, 0, 166),
        )
        draw.text(
            (x, y), sportsbox, font=font, fill="white",
            anchor="rt", stroke_width=stroke, stroke_fill="black",
        )
    return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)


def pause_plan(job: dict, pauses: list, pause_seconds: float) -> dict:
    if not math.isfinite(pause_seconds) or pause_seconds<=0:
        raise ValueError("停止時間は0秒より大きくしてください。")
    plan={}
    for code in dict.fromkeys(pauses):
        if code not in P_CODES:
            raise ValueError(f"不明な停止ポジションです：{code}")
        frame=job["p_frames"].get(code)
        if not isinstance(frame,int) or not 0<=frame<job["info"]["frames"]:
            raise ValueError(f"{code}のコマ番号が動画の範囲外です。コマを調整してください。")
        plan.setdefault(frame,[]).append(code)
    return plan


def export_video(job: dict, cfg: dict, pauses: list, pause_seconds: float, custom_positions: list, progress=None) -> Path:
    plan=pause_plan(job,pauses,pause_seconds)
    hold_frames=max(1,round(job["info"]["fps"]*pause_seconds))
    folder=Path(job["folder"])/"output";folder.mkdir(exist_ok=True)
    raw,final=folder/(uuid.uuid4().hex+"_raw.mp4"),folder/(uuid.uuid4().hex+"_analyzed.mp4")
    info=job["info"]
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*"mp4v"),info["fps"],(info["width"],info["height"]))
    if not writer.isOpened(): raise RuntimeError("動画の書き出しを開始できませんでした。")
    cap=cv2.VideoCapture(job["path"])
    completed=False
    try:
        for i in range(info["frames"]):
            ok,frame=cap.read()
            if not ok: raise ValueError("元動画の途中を読み取れませんでした。別の動画でお試しください。")
            image=draw_annotations(frame,primitives_for(job,i,cfg,None,custom_positions))
            writer.write(image)
            for code in plan.get(i,[]):
                paused=draw_annotations(frame,primitives_for(job,i,cfg,code,custom_positions),code,sportsbox_overlay(job,code))
                for _ in range(hold_frames):writer.write(paused)
            if progress and i%10==0:progress((i+1)/info["frames"],f"動画を書き出し中 {i+1}/{info['frames']}")
        completed=True
    finally:
        writer.release();cap.release()
        if not completed: raw.unlink(missing_ok=True)
    try:
        run_ffmpeg(["-i",raw,"-an","-c:v","libx264","-preset","veryfast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart","-threads","2",final])
        video_info(final)
        return final
    except Exception:
        final.unlink(missing_ok=True)
        raise
    finally:
        raw.unlink(missing_ok=True)


def white_frame_margins(frame: np.ndarray) -> tuple[int, int, int, int]:
    # Only trim a thin, nearly solid white border present on all four sides.
    # A bright wall or sky along one edge is part of the shot.
    white = np.all(frame >= 240, axis=2)
    rows = white.mean(axis=1) >= .995
    cols = white.mean(axis=0) >= .995
    top, bottom, left, right = (
        int(np.argmax(~edge)) for edge in (rows, rows[::-1], cols, cols[::-1])
    )
    if (0 < min(top, bottom, left, right)
            and max(top, bottom) <= frame.shape[0] * .08
            and max(left, right) <= frame.shape[1] * .08):
        return top, bottom, left, right
    return 0, 0, 0, 0


def export_images(job: dict,cfg: dict,codes: list,custom_positions: list) -> dict:
    if not codes: raise ValueError("切り出すポジションを1つ以上選んでください。")
    annotated_frames = []
    margins = []
    for code in codes:
        index = job["p_frames"][code]
        frame = read_frame(job["path"], index)
        margins.append(white_frame_margins(frame))
        annotated_frames.append(draw_annotations(
            frame, primitives_for(job, index, cfg, code, custom_positions),
            code, sportsbox_overlay(job, code),
        ))

    # Use the same crop on every frame so the reference lines still align.
    top, bottom, left, right = (min(edge) for edge in zip(*margins))
    h, w = annotated_frames[0].shape[:2]
    scale = min(1., 480 / max(h - top - bottom, w - left - right))
    cw = max(1, round((w - left - right) * scale))
    ch = max(1, round((h - top - bottom) * scale))
    cols = min(5, len(codes))
    rows = math.ceil(len(codes) / cols)
    sheet = np.zeros((rows * ch, cols * cw, 4), dtype=np.uint8)
    images = []
    zip_data = io.BytesIO()
    with zipfile.ZipFile(zip_data, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, (code, annotated) in enumerate(zip(codes, annotated_frames)):
            cropped = annotated[top:h - bottom, left:w - right]
            ok, png = cv2.imencode(".png", cropped)
            if not ok:
                raise RuntimeError("静止画を作成できませんでした。")
            data = png.tobytes()
            images.append((code, data))
            zf.writestr(code + ".png", data)
            x = k % cols * cw
            y = k // cols * ch
            sheet[y:y + ch, x:x + cw, :3] = cv2.resize(cropped, (cw, ch))
            sheet[y:y + ch, x:x + cw, 3] = 255
        ok, png = cv2.imencode(".png", sheet)
        if not ok:
            raise RuntimeError("連続写真を作成できませんでした。")
        sheet_data = png.tobytes()
        zf.writestr("sequence.png", sheet_data)
    return {"zip": zip_data.getvalue(), "sheet": sheet_data, "images": images}


def valid_point(value):
    return isinstance(value,list) and len(value)==2 and all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in value)


def accept_editor_event(job: dict,event: dict | None,phase: str,frame: int) -> bool:
    if not event or event.get("scene")!=f"{job['token']}:{phase}:{frame}" or not event.get("event_id") or event["event_id"]==job["last_event"]:
        return False
    # 編集データをiframeの正式な戻り値で受け取り、URLは操作しない。
    changed=False
    g=event.get("guides")
    if isinstance(g,dict):
        job["guides_manual"]=True
        for name in ["hip","vertical","back_vertical","top","bottom","radius"]:
            value=g.get(name)
            if isinstance(value,(float,int)) and math.isfinite(value) and 0<=value<=1:job["guides"][name]=value;changed=True
        if valid_point(g.get("head")):job["guides"]["head"]=g["head"];changed=True
        if isinstance(g.get("shaft"),list) and len(g["shaft"])==2 and all(valid_point(p) for p in g["shaft"]):job["guides"]["shaft"]=g["shaft"];changed=True
        if isinstance(g.get("ankles"),list) and len(g["ankles"])==2 and all(isinstance(v,(float,int)) and math.isfinite(v) and 0<=v<=1 for v in g["ankles"]):job["guides"]["ankles"]=g["ankles"];changed=True
    body=event.get("body")
    if isinstance(body,dict) and all(valid_point(body.get(k)) for k in ["ls","rs","sh","hip"]):
        baseline=get_pose(job,frame,phase) or {}
        updated = {k: body[k] for k in ("ls", "rs", "sh", "hip")}
        points = ("rk", *RIGHT_LEG_POINTS) if phase == "P1" else ("rk",)
        for key in points:
            if valid_point(body.get(key)):
                updated[key] = body[key]
        job["body_overrides"][phase] = {**baseline, **updated}
        changed = True
    shapes=event.get("shapes")
    if isinstance(shapes,list) and len(shapes)<=250:
        good=[]
        for shape in shapes:
            if not isinstance(shape,dict) or shape.get("type") not in ["line","circle","free"]:continue
            if not re.fullmatch(r"#[0-9a-fA-F]{6}",str(shape.get("color",""))):continue
            if not isinstance(shape.get("width"),(int,float)) or not 1<=shape["width"]<=10:continue
            if shape.get("target_p") not in [None,*P_CODES]:continue
            if shape["type"]=="line" and not(valid_point(shape.get("a")) and valid_point(shape.get("b"))):continue
            if shape["type"]=="circle" and not(valid_point(shape.get("center")) and isinstance(shape.get("radius"),(float,int)) and math.isfinite(shape["radius"]) and 0<shape["radius"]<=2):continue
            if shape["type"]=="free" and not(isinstance(shape.get("points"),list) and 2<=len(shape["points"])<=5000 and all(valid_point(p) for p in shape["points"])):continue
            good.append(shape)
        job["shapes"]=good;changed=True
    job["last_event"]=event["event_id"]
    if changed:
        job["output_video"]=None;job["output_images"]=None
    return changed


def editor_phase_change(job: dict,event,phase: str,frame: int) -> str | None:
    if not isinstance(event,dict) or event.get("action")!="select_phase":return None
    target=event.get("phase")
    if target not in P_CODES or event.get("scene")!=f"{job['token']}:{phase}:{frame}":return None
    if not event.get("event_id") or event["event_id"]==job["last_event"]:return None
    accept_editor_event(job,event,phase,frame)
    return target


EDITOR_HTML = r"""
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box}body{margin:0;font-family:Arial,"Yu Gothic",Meiryo,sans-serif;color:#173831}#root{padding:0 1px 8px}html,body{overscroll-behavior:none}canvas{-webkit-touch-callout:none;user-select:none}.toolbar{margin-top:6px;display:flex;gap:5px;align-items:center;flex-wrap:wrap;padding:8px;background:#edf2ec;border:1px solid #d8e0d5;border-radius:7px}button,select,input{font:inherit}button{min-height:36px;border:1px solid #b7c7b1;border-radius:5px;background:#fff;color:#173831;padding:6px 10px;cursor:pointer;font-size:12px}button.active{background:#1c4c40;color:white;border-color:#1c4c40}.toolbar label{font-size:11px;display:flex;align-items:center;gap:3px}.toolbar input[type=color]{width:29px;height:31px;padding:0;border:0;background:none}.toolbar select{height:31px;border:1px solid #b7c7b1;border-radius:4px;background:white;font-size:11px}.canvas-wrap{position:relative;display:inline-flex;max-width:100%}.phase-overlay{position:absolute;top:8px;left:8px;z-index:2;padding:3px;background:rgba(255,255,255,.9);border:1px solid #a9bbaa;border-radius:7px;box-shadow:0 1px 6px #0002}.phase-overlay select{font-size:16px;font-weight:bold;min-height:42px;min-width:84px;padding:6px 10px;background:transparent;border:0;border-radius:4px;color:#173831}.view{margin-top:0;background:#e9ede6;border:1px solid #cbd5c6;border-radius:7px;display:flex;justify-content:center;align-items:center;overflow:hidden}canvas{display:block;max-width:100%;max-height:620px;width:auto;height:auto;touch-action:none;cursor:grab}.hint{font-size:11px;line-height:1.8;color:#65755e;margin:7px 2px 0}#saved{font-size:11px;min-height:20px;margin:5px 2px 0;color:#36734a}button:focus-visible{outline:3px solid #c8dc8e}@media(max-width:600px){.toolbar{padding:6px;gap:4px}button{font-size:11px;min-height:38px;padding:5px 8px}canvas{max-height:560px}}
</style></head><body><div id="root"><div class="view"><div class="canvas-wrap"><canvas id="canvas" width="720" height="960" aria-label="管理線を直接ドラッグするプレビュー"></canvas><div class="phase-overlay"><select id="phaseSelect" aria-label="ポジション"><option value="P1">P1</option><option value="P2">P2</option><option value="P3">P3</option><option value="P4">P4</option><option value="P5">P5</option><option value="P6">P6</option><option value="P7">P7</option><option value="P8">P8</option><option value="P9">P9</option><option value="P10">P10</option></select></div></div></div><div class="toolbar"><button data-tool="move" class="active">✥ 線を動かす</button><button data-tool="line">╱ 直線</button><button data-tool="circle">○ 円</button><button data-tool="free">〰 手描き</button><input id="color" type="color" value="#ffff00" aria-label="手描きの色"><select id="width" aria-label="手描きの太さ"><option>2</option><option selected>3</option><option>4</option><option>5</option><option>6</option></select><select id="dash" aria-label="手描きの線種"><option value="solid">実線</option><option value="dashed">点線</option></select><label><input id="angle" type="checkbox" checked>角度</label><label><input id="bind" type="checkbox" checked>このPのみ</label><button id="undo">↶ 戻す</button><button id="clear">描画を消去</button></div><p class="hint">丸をドラッグすると位置・角度を調整できます　シャフトとヒップは線そのものも動かせます<br>直線・円・手描きは映像上をドラッグして追加します　3点を順番に押す操作はありません</p><div id="saved" role="status">動画と管理線を準備しています…</div></div><script>
const $=id=>document.getElementById(id),canvas=$('canvas'),ctx=canvas.getContext('2d');let args=null,scene=null,g=null,body=null,shapes=[],tool='move',drag=null,active=null,history=[],lastEvent=null,lastHeight=0;const img=new Image();const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v)),clone=x=>JSON.parse(JSON.stringify(x));let changedG=false,changedB=false,changedS=false;
function message(type,extra={}){window.parent.postMessage({isStreamlitMessage:true,type,...extra},'*');}
function height(){const h=Math.ceil($('root').getBoundingClientRect().height)+8;if(h!==lastHeight){lastHeight=h;message('streamlit:setFrameHeight',{height:h});}}
function visible(s){return s.target_p?s.target_p===args.phase:args.custom_visible!==false;}
function style(s){ctx.strokeStyle=s.color||'#ffff00';ctx.lineWidth=Math.max(1,(s.width||3)*canvas.height/1080);ctx.lineCap=s.dash==='dashed'?'butt':'round';ctx.lineJoin='round';const dash=Math.max(2,4*canvas.height/1080);ctx.setLineDash(s.dash==='dashed'?[dash,dash]:[]);}
function text(label,p,color,angle=false){ctx.save();ctx.font=`bold ${angle?Math.max(14,canvas.height*26/1080):Math.max(18,canvas.height*.03)}px sans-serif`;ctx.lineWidth=angle?2:3;ctx.strokeStyle='#101c17';ctx.fillStyle=color;ctx.setLineDash([]);const x=clamp(p[0],0,.88)*canvas.width,y=clamp(p[1],.03,.97)*canvas.height;ctx.strokeText(label,x,y);ctx.fillText(label,x,y);ctx.restore();}
function geometry(item){const s=clone(item);if(s.id==='shaft'){const[a,b]=g.shaft;s.a=a;s.b=[a[0]+(b[0]-a[0])*2.2,a[1]+(b[1]-a[1])*2.2];s.label_pos=[a[0]+(b[0]-a[0])*.45+.014,a[1]+(b[1]-a[1])*.45+.015];}if(s.id==='hip'){s.a=[g.hip,g.top];s.b=[g.hip,g.bottom];}if(s.id==='vertical'){s.a=[g.vertical,g.top];s.b=[g.vertical,g.bottom];}if(s.id==='back_vertical'){s.a[0]=s.b[0]=g.back_vertical;}if(s.id==='ankles'){s.a=[g.ankles[s.part],g.top];s.b=[g.ankles[s.part],g.bottom];}if(s.id==='head_reference'){s.center=g.head;s.radius=g.radius;}if(s.id==='head'){const d=s.head_delta||[0,0];s.center=[g.head[0]+d[0],g.head[1]+d[1]];s.radius=g.radius;}if(body){if(s.id==='shoulders'){s.a=body.ls;s.b=body.rs;}if(s.id==='hip_right_knee'&&body.rk){s.a=body.hip;s.b=body.rk;}if(s.id==='right_leg_outer'&&args.phase==='P1'){s.a=body.right_ankle_outer;s.b=body.right_thigh_outer;}if(s.id==='tilt_zero'){s.a=body.hip;s.b=[body.hip[0],body.sh[1]];}if(s.id==='spine'||s.id==='tilt'){s.a=body.hip;s.b=body.sh;}}if(s.id==='shoulders'){const right=s.a[0]>=s.b[0]?s.a:s.b;s.label_pos=[right[0]+.025,right[1]-.01];}return s;}
function draw(s){ctx.save();style(s);ctx.beginPath();if(s.type==='line'){ctx.moveTo(s.a[0]*canvas.width,s.a[1]*canvas.height);ctx.lineTo(s.b[0]*canvas.width,s.b[1]*canvas.height);}else if(s.type==='circle'&&s.center){ctx.arc(s.center[0]*canvas.width,s.center[1]*canvas.height,s.radius*canvas.width,0,Math.PI*2);}else if(s.type==='free'){s.points.forEach((p,i)=>i?ctx.lineTo(p[0]*canvas.width,p[1]*canvas.height):ctx.moveTo(p[0]*canvas.width,p[1]*canvas.height));}ctx.stroke();ctx.restore();if(s.type==='line'&&s.angle){let a=Math.round(Math.atan2(Math.abs((s.b[1]-s.a[1])*canvas.height),Math.abs((s.b[0]-s.a[0])*canvas.width))*180/Math.PI);if(s.vertical){const dx=(s.b[0]-s.a[0])*canvas.width,dy=(s.b[1]-s.a[1])*canvas.height;a=Math.round(Math.atan2(Math.abs(dx),Math.abs(dy))*180/Math.PI);if(dx*(s.target_dir||-1)<0)a=-a;}text(a+'°',s.label_pos||[(s.a[0]+s.b[0])/2+.014,(s.a[1]+s.b[1])/2-.01],s.color,true);}}
function handles(){const out=[],shown=id=>args.primitives.some(s=>s.id===id);const add=(p,type,key,extra={})=>{if(p)out.push({p,type,key,...extra});};if(shown('shaft'))g.shaft.forEach((p,i)=>add(p,'shaft',i,{label:i===0?'ヘッド側':'グリップ側',color:i===0?'#58d6ef':'#e9f38a'}));if(shown('hip'))add([g.hip,(g.top+g.bottom)/2],'hip',null,{label:'ヒップ',color:'#ffbf69'});
if(args.active_line==='head'&&shown('head_reference')&&g.head){add(g.head,'head');add([g.head[0]+g.radius,g.head[1]],'radius');}if(args.active_line==='vertical'&&shown('vertical'))add([g.vertical,(g.top+g.bottom)/2],'vertical');if(args.active_line==='back_vertical'&&shown('back_vertical')){const s=args.primitives.find(p=>p.id==='back_vertical');add([g.back_vertical,(s.a[1]+s.b[1])/2],'back_vertical');}if(args.active_line==='ankles'&&shown('ankles'))g.ankles.forEach((x,i)=>add([x,(g.top+g.bottom)/2],'ankles',i));
if(body){if(['spine','tilt'].includes(args.active_line)&&shown(args.active_line)){add(body.sh,'body','sh');add(body.hip,'body','hip');}if(args.active_line==='shoulders'&&shown('shoulders')){add(body.ls,'body','ls');add(body.rs,'body','rs');}if(args.active_line==='hip_right_knee'&&shown('hip_right_knee')){add(body.hip,'body','hip');add(body.rk,'body','rk');}if(args.phase==='P1'&&args.active_line==='right_leg_outer'&&shown('right_leg_outer')){add(body.right_ankle_outer,'body','right_ankle_outer');add(body.right_thigh_outer,'body','right_thigh_outer');}}
shapes.forEach((s,i)=>{if(!visible(s))return;if(s.type==='line'){add(s.a,'custom',i,{end:'a'});add(s.b,'custom',i,{end:'b'});}if(s.type==='circle'){add(s.center,'custom',i,{end:'center'});add([s.center[0]+s.radius,s.center[1]],'custom',i,{end:'radius'});}if(s.type==='free'&&s.points.length)add(s.points[0],'custom',i,{end:'free'});});return out;}
function drawSportsbox(label){
    if(!label)return;
    const size=Math.max(8,Math.min(Math.max(15,Math.round(canvas.height*.028)),Math.floor(canvas.width/17)));
    const margin=Math.max(6,Math.round(size*.65));
    const pad=Math.max(4,Math.round(size*.35)),stroke=Math.max(1,Math.round(size*.1));
    ctx.save();
    ctx.font=`bold ${size}px "Yu Gothic", Meiryo, sans-serif`;
    ctx.textAlign='right';
    ctx.textBaseline='alphabetic';
    const metrics=ctx.measureText(label);
    const x=canvas.width-margin-pad-stroke-metrics.actualBoundingBoxRight;
    const y=margin+pad+stroke+metrics.actualBoundingBoxAscent;
    const left=x-metrics.actualBoundingBoxLeft-stroke-pad;
    const bottom=y+metrics.actualBoundingBoxDescent+stroke+pad;
    ctx.fillStyle='rgba(0,0,0,.65)';
    ctx.fillRect(left,margin,canvas.width-margin-left,bottom-margin);
    ctx.setLineDash([]);
    ctx.lineJoin='round';
    ctx.lineWidth=2*stroke;
    ctx.strokeStyle='#000';
    ctx.fillStyle='#fff';
    ctx.strokeText(label,x,y);
    ctx.fillText(label,x,y);
    ctx.restore();
}
function render(){if(!args||!img.complete||!img.naturalWidth)return;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0,canvas.width,canvas.height);args.primitives.forEach(s=>draw(geometry(s)));shapes.filter(visible).forEach(draw);if(active)draw(active);if(tool==='move'){for(const h of handles()){ctx.save();ctx.setLineDash([]);ctx.fillStyle=h.color||'#e5f28b';ctx.strokeStyle='#163c30';ctx.lineWidth=2;ctx.beginPath();ctx.arc(h.p[0]*canvas.width,h.p[1]*canvas.height,Math.max(8,canvas.height*.012),0,Math.PI*2);ctx.fill();ctx.stroke();ctx.restore();if(h.label)text(h.label,[h.p[0]+.018,h.p[1]-.015],h.color);}}drawSportsbox(args.sportsbox);height();}
function remember(){history.push(clone({g,body,shapes}));if(history.length>25)history.shift();}
function emit(){if(!changedG&&!changedB&&!changedS)return;$('phaseSelect').disabled=true;lastEvent=Date.now().toString(36)+'-'+Math.random().toString(36).slice(2);message('streamlit:setComponentValue',{dataType:'json',value:{scene,event_id:lastEvent,guides:changedG?g:null,body:changedB?body:null,shapes:changedS?shapes:null}});changedG=changedB=changedS=false;$('saved').textContent='変更を反映中…';}
function pos(e){const r=canvas.getBoundingClientRect();return[clamp((e.clientX-r.left)/r.width),clamp((e.clientY-r.top)/r.height)];}
function hit(p){const rect=canvas.getBoundingClientRect();const distances=handles().map(h=>({h,d:Math.hypot((h.p[0]-p[0])*rect.width,(h.p[1]-p[1])*rect.height)})).filter(v=>v.d<25).sort((a,b)=>a.d-b.d);if(distances.length)return distances[0].h;
const candidates=[];if(args.primitives.some(s=>s.id==='hip'))candidates.push({type:'hip',d:Math.abs(g.hip-p[0])*rect.width});if(args.primitives.some(s=>s.id==='shaft')){const[a,b]=g.shaft,dx=(b[0]-a[0])*2.2*rect.width,dy=(b[1]-a[1])*2.2*rect.height,x=(p[0]-a[0])*rect.width,y=(p[1]-a[1])*rect.height,t=clamp((x*dx+y*dy)/(dx*dx+dy*dy||1));candidates.push({type:'shaftMove',d:Math.hypot(x-dx*t,y-dy*t)});}return candidates.filter(v=>v.d<13).sort((a,b)=>a.d-b.d)[0]||null;}
function translate(pts,delta){const dx=clamp(delta[0],-Math.min(...pts.map(p=>p[0])),1-Math.max(...pts.map(p=>p[0]))),dy=clamp(delta[1],-Math.min(...pts.map(p=>p[1])),1-Math.max(...pts.map(p=>p[1])));return pts.map(p=>[p[0]+dx,p[1]+dy]);}
let activePointer=null;
canvas.addEventListener('pointerdown',e=>{if(!args||args.disabled||activePointer!==null||!e.isPrimary||e.button!==0)return;activePointer=e.pointerId;const p=pos(e);canvas.setPointerCapture(e.pointerId);if(tool==='move'){drag=hit(p);if(drag){remember();drag.last=p;$('saved').textContent='調整中…';}}else{remember();const base={color:$('color').value,width:Number($('width').value),dash:$('dash').value,target_p:$('bind').checked?args.phase:null};if(tool==='line')active={type:'line',a:p,b:[...p],angle:$('angle').checked,...base};if(tool==='circle')active={type:'circle',center:p,radius:.001,...base};if(tool==='free')active={type:'free',points:[p],...base};}e.preventDefault();});
canvas.addEventListener('pointermove',e=>{if(e.pointerId!==activePointer||(!drag&&!active))return;const p=pos(e);if(drag){const d=drag;if(d.type==='shaft'){g.shaft[d.key]=p;changedG=true;}else if(d.type==='shaftMove'){g.shaft=translate(g.shaft,[p[0]-d.last[0],p[1]-d.last[1]]);d.last=p;changedG=true;}else if(d.type==='hip'){g.hip=p[0];changedG=true;}else if(d.type==='vertical'||d.type==='back_vertical'){g[d.type]=p[0];changedG=true;}else if(d.type==='ankles'){g.ankles[d.key]=p[0];changedG=true;}else if(d.type==='head'){g.head=p;changedG=true;}else if(d.type==='radius'){g.radius=clamp(Math.hypot((p[0]-g.head[0])*canvas.width,(p[1]-g.head[1])*canvas.height)/canvas.width,.01,.25);changedG=true;}else if(d.type==='body'){if(d.key==='sh'){const delta=[p[0]-body.sh[0],p[1]-body.sh[1]];body.ls=[clamp(body.ls[0]+delta[0]),clamp(body.ls[1]+delta[1])];body.rs=[clamp(body.rs[0]+delta[0]),clamp(body.rs[1]+delta[1])];}body[d.key]=p;if(['ls','rs'].includes(d.key))body.sh=[(body.ls[0]+body.rs[0])/2,(body.ls[1]+body.rs[1])/2];changedB=true;}else if(d.type==='custom'){const s=shapes[d.key];if(d.end==='radius')s.radius=clamp(Math.hypot((p[0]-s.center[0])*canvas.width,(p[1]-s.center[1])*canvas.height)/canvas.width,.003,2);else if(d.end==='free'){s.points=translate(s.points,[p[0]-d.last[0],p[1]-d.last[1]]);d.last=p;}else s[d.end]=p;changedS=true;}}
if(active){if(active.type==='line')active.b=p;if(active.type==='circle')active.radius=Math.max(.001,Math.hypot((p[0]-active.center[0])*canvas.width,(p[1]-active.center[1])*canvas.height)/canvas.width);if(active.type==='free'&&active.points.length<5000)active.points.push(p);}render();e.preventDefault();});
function finishGesture(e){if(e.pointerId!==activePointer)return;activePointer=null;if(active){const valid=active.type==='free'?active.points.length>=2:active.type==='circle'?active.radius>.003:Math.hypot(active.a[0]-active.b[0],active.a[1]-active.b[1])>.005;if(valid){shapes.push(active);changedS=true;}active=null;}drag=null;render();emit();}
for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,finishGesture);
document.querySelectorAll('[data-tool]').forEach(b=>b.onclick=()=>{tool=b.dataset.tool;document.querySelectorAll('[data-tool]').forEach(x=>x.classList.toggle('active',x===b));canvas.style.cursor=tool==='move'?'grab':'crosshair';render();});
$('undo').onclick=()=>{if(!history.length)return;const prev=history.pop();g=prev.g;body=prev.body;shapes=prev.shapes;changedG=changedS=true;changedB=!!body;render();emit();};
$('clear').onclick=()=>{remember();shapes=shapes.filter(s=>!visible(s));changedS=true;render();emit();};
$('phaseSelect').onchange=()=>{if(!args)return;lastEvent=Date.now().toString(36)+'-'+Math.random().toString(36).slice(2);$('phaseSelect').disabled=true;message('streamlit:setComponentValue',{dataType:'json',value:{scene,event_id:lastEvent,action:'select_phase',phase:$('phaseSelect').value}});};
let scrolledView=null;
function returnToEditor(id){
    if(!id||scrolledView===id)return;
    scrolledView=id;
    requestAnimationFrame(()=>{
        if(!window.frameElement)return;
        try{
            if(sessionStorage.getItem('golf-restored-view')===id)return;
            sessionStorage.setItem('golf-restored-view',id);
        }catch{}
        window.frameElement.scrollIntoView({block:'start'});
    });
}
img.onload=()=>{render();height();};window.addEventListener('message',e=>{if(e.source!==window.parent||e.data.type!=='streamlit:render')return;args=e.data.args;$('phaseSelect').value=args.phase;$('phaseSelect').disabled=!!(lastEvent&&args.ack!==lastEvent);returnToEditor(args.restore_view);const newScene=args.scene!==scene;scene=args.scene;g=clone(args.guides);body=clone(args.body);shapes=clone(args.shapes);if(newScene){history=[];tool='move';document.querySelectorAll('[data-tool]').forEach(b=>b.classList.toggle('active',b.dataset.tool==='move'));}canvas.width=args.width;canvas.height=args.height;const src='data:image/jpeg;base64,'+args.image;if(img.src!==src)img.src=src;else render();$('saved').textContent=lastEvent&&args.ack===lastEvent?'✓ 変更を保存しました':'丸・線を直接ドラッグして調整してください';height();});
new ResizeObserver(height).observe($('root'));window.addEventListener('resize',height);message('streamlit:componentReady',{apiVersion:1});height();
</script></body></html>
"""


def editor_component():
    folder=Path(tempfile.gettempdir())/("golf_simple_editor_"+hashlib.sha256(EDITOR_HTML.encode()).hexdigest()[:16])
    folder.mkdir(exist_ok=True)
    index=folder/"index.html"
    if not index.exists():index.write_text(EDITOR_HTML,encoding="utf-8")
    return declare_component("golf_simple_drag_editor",path=str(folder))


MOBILE_UPLOAD_HTML = r"""<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{margin:0;font-family:Arial,'Yu Gothic',sans-serif;color:#19382c}*{box-sizing:border-box}.box{padding:12px;background:#f0f4ef;border:1px solid #cbd5c6;border-radius:8px}button{font:inherit;padding:12px;border:1px solid #9caf98;border-radius:6px;background:white;color:#19382c;cursor:pointer;margin:4px 4px 4px 0}button:disabled{opacity:.45;cursor:default}#send{background:#247052;color:white}#name{margin:10px 0;overflow-wrap:anywhere;font-weight:bold}#status{font-size:14px;line-height:1.6;white-space:pre-wrap}progress{width:100%;height:18px}small{display:block;margin:8px 0;color:#52644c;line-height:1.6}
</style><div class="box" id="root"><button id="choose">① ファイルから__MEDIA__を選ぶ</button><button id="gallery">__GALLERY__</button><input type="file" id="files" hidden><input type="file" id="videos" accept="__ACCEPT__" hidden><div id="name">__MEDIA__はまだ選ばれていません</div><button id="send" disabled>② __MEDIA__を送信</button><progress id="progress" value="0" max="100"></progress><div id="status" role="status">端末内に保存した__MEDIA__を選んでください（__LIMIT__MBまで）</div><small>__MEDIA__名が出たら「② __MEDIA__を送信」を押してください__AFTER_SEND__</small></div><script>
const $=id=>document.getElementById(id);let file=null,busy=false,args={},pending=null,picker=false;
function post(type,extra={}){window.parent.postMessage({isStreamlitMessage:true,type,...extra},'*');}
function height(){post('streamlit:setFrameHeight',{height:Math.ceil($('root').getBoundingClientRect().height)+8});}
function status(t){$('status').textContent=t;height();}
function controls(){for(const id of ['choose','gallery','files','videos'])$(id).disabled=busy;$('send').disabled=busy||!file;}
for(const [button,input] of [['choose','files'],['gallery','videos']]){$(button).onclick=()=>{picker=true;$(input).value='';$(input).click();};$(input).onchange=()=>{picker=false;const f=$(input).files[0];if(!f)return;file=f;$('name').textContent=f.name+'（'+(f.size/1048576).toFixed(1)+' MB）';$('progress').value=0;if(!f.size||f.size>__LIMIT__*1048576){file=null;status('空の__MEDIA__、または__LIMIT__MBを超える__MEDIA__です。別の__MEDIA__を選んでください。');}else status('__MEDIA__を選択しました。「② __MEDIA__を送信」を押してください。');controls();height();};}
window.addEventListener('focus',()=>{if(picker)setTimeout(()=>{if(picker){picker=false;status('__MEDIA__名がまだ返っていません。「① ファイルから__MEDIA__を選ぶ」で端末内の__MEDIA__を選んでください。');}},2000);});
function send(value,expected){return new Promise((resolve,reject)=>{let attempts=0;const transmit=()=>{post('streamlit:setComponentValue',{dataType:'json',value});};const timer=setInterval(()=>{if(++attempts>=3){clearInterval(timer);pending=null;reject(Error('応答がありません。アプリを開き直してから、もう一度送信してください。'));}else{status('接続を確認して再送しています…');transmit();}},15000);pending={id:value.id,expected,resolve:()=>{clearInterval(timer);pending=null;resolve();},reject:e=>{clearInterval(timer);pending=null;reject(Error(e));}};transmit();});}
function read(slice){return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);r.onerror=()=>reject(Error('この__MEDIA__を端末から読み出せません。端末内へ保存した__MEDIA__で試してください。'));r.readAsDataURL(slice);});}
$('send').onclick=async()=>{if(!file||busy)return;const chosen=file,id=Date.now().toString(36)+'-'+Math.random().toString(36).slice(2);busy=true;controls();try{status('送信を開始しています…');await send({kind:'begin',id,name:chosen.name,size:chosen.size},0);let seq=0;for(let offset=0;offset<chosen.size;offset+=262144){const end=Math.min(chosen.size,offset+262144),data=await read(chosen.slice(offset,end));status('__MEDIA__を送信中… '+Math.floor(offset/chosen.size*100)+'%');await send({kind:'chunk',id,seq,data},seq+1);seq++;$('progress').value=Math.round(end/chosen.size*100);}status('受信完了__AFTER_SEND__');}catch(e){status(e.message);}finally{busy=false;controls();}};
window.addEventListener('message',e=>{if(e.source!==window.parent||e.data.type!=='streamlit:render')return;args=e.data.args||{};if(pending&&args.transfer_id===pending.id){if(args.error)pending.reject(args.error);else if(args.next_seq>=pending.expected)pending.resolve();}if(!file&&args.complete){$('name').textContent='受信済み：'+args.name;$('progress').value=100;status('受信完了__AFTER_SEND__');}height();});
new ResizeObserver(height).observe($('root'));post('streamlit:componentReady',{apiVersion:1});height();
</script></html>"""


def accept_mobile_upload(state: dict, event, size_limit_mb: int) -> bool:
    if not isinstance(event,dict):return False
    transfer_id=event.get("id")
    if not isinstance(transfer_id,str) or not re.fullmatch(r"[A-Za-z0-9-]{1,80}",transfer_id):return False
    kind=event.get("kind")
    if kind=="begin":
        if state.get("id")==transfer_id:return False
        size=event.get("size");name=event.get("name")
        if not isinstance(size,int) or isinstance(size,bool) or not 0<size<=size_limit_mb*1024*1024:return False
        if not isinstance(name,str) or not 0<len(name)<=512:return False
        state.clear();state.update(id=transfer_id,name=name,size=size,data=bytearray(),next_seq=0,complete=False,error=None)
        return True
    if kind!="chunk" or state.get("id")!=transfer_id or state.get("error"):return False
    seq=event.get("seq")
    if not isinstance(seq,int) or isinstance(seq,bool) or seq!=state.get("next_seq"):return False
    encoded=event.get("data")
    try:
        if not isinstance(encoded,str) or len(encoded)>349528:raise ValueError("送信データのサイズが不正です。")
        chunk=base64.b64decode(encoded,validate=True)
        expected=min(262144,state["size"]-len(state["data"]))
        if not expected or len(chunk)!=expected:raise ValueError("ファイルの一部を受信できませんでした。もう一度ファイルを選んで送信してください。")
        state["data"].extend(chunk);state["next_seq"]+=1
        state["complete"]=len(state["data"])==state["size"]
    except (ValueError,TypeError) as exc:state["error"]=str(exc)
    return True


class MobileUpload:
    def __init__(self, state: dict):
        self.name = state["name"]
        self.size = state["size"]
        self.file_id = state["id"]
        self.data = state["data"]

    def getbuffer(self):
        return memoryview(self.data)


def mobile_upload(kind: str, state: dict, key: str) -> MobileUpload | None:
    image = kind == "image"
    limit = 15 if image else 200
    media = "画像" if image else "動画"
    html = (MOBILE_UPLOAD_HTML.replace("__MEDIA__", media)
            .replace("__LIMIT__", str(limit))
            .replace("__GALLERY__", "写真から選ぶ" if image else "写真・動画から選ぶ")
            .replace("__ACCEPT__", "image/*" if image else "video/*")
            .replace("__AFTER_SEND__", "。下に読み取り結果が表示されます。" if image else "。下の「動画を読み込んで解析」を押してください。"))
    folder = Path(tempfile.gettempdir()) / ("golf_mobile_upload_" + hashlib.sha256(html.encode()).hexdigest()[:16])
    folder.mkdir(exist_ok=True)
    index = folder / "index.html"
    if not index.exists():
        index.write_text(html, encoding="utf-8")
    component = declare_component(f"golf_mobile_upload_{kind}", path=str(folder))
    event = component(
        transfer_id=state.get("id"), next_seq=state.get("next_seq", -1),
        complete=state.get("complete", False), error=state.get("error"),
        name=state.get("name", ""), key=key, default=None,
    )
    if accept_mobile_upload(state, event, limit):
        st.rerun()
    if state.get("error"):
        st.error(state["error"])
    return MobileUpload(state) if state.get("complete") and not state.get("error") else None


def clear_outputs():
    if "job" in st.session_state:
        st.session_state["job"]["output_video"]=None;st.session_state["job"]["output_images"]=None


def switch_mode(front=False):
    st.session_state["camera_mode"]="front" if front else "back"
    st.session_state["enabled_lines"]=(FRONT_DEFAULT_LINES if front else DEFAULT_LINES)[:]
    clear_outputs()


def frame_change(code,widget):
    job=st.session_state["job"];job["p_frames"][code]=int(st.session_state[widget]);job["body_overrides"].pop(code,None)
    if code=="P1" and not job.get("guides_manual"):
        ref=job["poses"][job["p_frames"][code]]
        if ref:job["guides"]=initial_guides(ref,job["info"])
    clear_outputs()


def step_frame(code,widget,delta):
    job=st.session_state["job"];st.session_state[widget]=max(0,min(job["info"]["frames"]-1,int(st.session_state[widget])+delta));frame_change(code,widget)


def refresh_pose_points(job: dict, keys: tuple, progress) -> int:
    detected = analyze_skeleton(Path(job["path"]), job["info"], progress)
    updates = 0
    for index, new_pose in enumerate(detected[:len(job["poses"])]):
        if new_pose and all(valid_point(new_pose.get(key)) for key in keys):
            pose = job["poses"][index] or new_pose
            job["poses"][index] = {**pose, **{key: new_pose[key] for key in keys}}
            updates += 1
    for code, body in job["body_overrides"].items():
        pose = job["poses"][job["p_frames"][code]]
        if pose:
            for key in keys:
                if key in pose:
                    body.setdefault(key, pose[key])
    if updates:
        clear_outputs()
    return updates


# Sportsbox Turn screenshot recognition.
SPORTSBOX_POSITIONS = {"P1": "Address", "P4": "TOP", "P7": "IMPACT"}
SPORTSBOX_METRICS = {"chest": "胸", "pelvis": "骨盤"}
SPORTSBOX_REFERENCE_SIZE = (2048, 946)
# Address, Top, and the first card immediately after DAH (Impact).
# Coordinates are for the reference Turn screen, excluding the LIVE column.
SPORTSBOX_ROWS = {
    "chest": (355, [(639, 929), (1260, 1550), (1881, 2171)], 274),
    "pelvis": (722, [(630, 920), (1252, 1542), (1873, 2163)], 641),
}
SPORTSBOX_BADGES = {
    "chest": [(802, 280, 914, 310), (1475, 280, 1536, 310), (1786, 280, 1846, 310)],
    "pelvis": [(794, 647, 906, 677), (1467, 647, 1528, 677), (1778, 647, 1838, 677)],
}
SPORTSBOX_RESAMPLE = Image.Resampling.LANCZOS


# Packed reference masks for digits, metric titles, and position badges.
SPORTSBOX_GLYPH_TEMPLATES = {'digits': {'0': ['03c01ff83ffc7ffe7c3e781ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff81f7c3e7ffe3ffc1ff807e0'],
            '1': ['01ff1ffffffffffffffffcffe0ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff'],
            '2': ['03c01ff83ffc7ffe7e3ef81ff80ff00f000f001f003f007e00fc01f803f007e00fc01f803f00ff00ffffffffffffffff',
                  '03c01ff83ffc7ffe7e7ef81ef80ff00f000f001f003e007e00fc01f803f007e00fc01f807f00ff00ffffffffffffffff',
                  '01800ff83ffc7ffe7e7e781ff80ff00f000f001f001f003e007e01fc03f807e00fc01f803f007e00ffffffffffffffff'],
            '3': ['7ffe7ffe7ffe7ffe003e007c00f801f007e00ff00ffc07fe03fe003f001f001f001f181ff81ffc3e7ffe3ffc1ff807e0',
                  '7ffe7ffe7ffe7ffe003e007c01f803f007e00ff007f807fc02fe003f001f001f001f301ff83efc7e7ffc3ffc1ff00380',
                  '7ffe7ffe7ffe7ffe003e007c01f803f007e00ff00ff807fc00fe003e001e001f001f301ef83efc7e7ffc3ffc1ff00380',
                  '7ffe7ffe7ffe7ffe003e007c01f803f007e00ff007f807fc02fe003f001f001f001f301ff83efc7e7ffc3ffc1ff00380',
                  '7ffe7ffe7ffe7ffe003e007c00f803f007e00ff007fc07fe02fe001f001f001f001f101ff83ffc7e7ffe3ffc1ff803c0'],
            '4': ['01f003e003e007c007c007800f800f801f001f003e3c3e3c7c3c7c3cf83cf83cffffffffffffffff003c003c003c003c',
                  '01e003e003e007c007c00f800f800f001f001e003e3c3e3c7c3c7c3cf83cf83cffffffffffffffff003c003c003c003c'],
            '5': ['3ffc3ffc3ffc7ffc780078007800780078007ff07ffc7ffe7ffe083f001f001f001f101ff81ffc3e7ffe3ffc1ff807e0',
                  '7ffc7ffc7ffc7ffc780078007800780078807ff07ff87ffc7cfe003e001e001f001f301ff83efc7e7ffc3ffc1ff007c0'],
            '6': ['01800ff83ffc3ff87c30f800f000f000f000f3f0fff8fffcfffef83ef81ef01ff00ff01ef81e7c3e7ffc3ffc1ff003c0'],
            '7': ['ffffffffffffffff001f003e003e007c007c00f800f800f801f001f003e003e007e007c007c00f800f801f001f001f00'],
            '8': ['07e01ff83ffc7ffe7e7e7c1f7c1ef81e7c3e7ffe3ffc3ffc7ffe7e7efc1ff81ff80ff80ff81f7c3f7ffe3ffc1ff80ff0'],
            '9': ['01800ff03ffc3ffc7e3e781ef81ff00ff00ff81f781f7fff3fff3fff0fcf000f000f001f001f0c3e1ffc3ffc1ff007c0']},
 'chest': '3c40000003f800007e40000183f800007e40000183f80000c340000180400000c37c3873e04625aec07e7cfbe04627bf807e7cf9c046273ec06644c180462633c0667ee1804626338066fe7180462631c066e03980462633c366c01980462631c36644c98046663166666cd9804366317e667cf8e043e6313c663870e043e631',
 'chest_Address': ['00000000000431f1e3e7cf1e31f1f3e7cf1f31913236199379991236199379891a361810498912378e1c49891bf7cf1ec9891be6878f498913e6038779891be60181f99912661991fd9132661993c5f1f237cf9f85f1e227cf1e000000000604',
                   '00000000040431f1e3e7cf1e31f1e3e7cf1e31913236199379991236199179891a361810498912378c1c49891bf7cf1ec9891be6878f498913e6038779891be60181f99912641991cd9132661993c5f1f237cf9f85f1e227cf1e000000000604',
                   '00000000060c31e1e3c7cf1e31f1e3e7cf1e31913336099379991236189179991a361810498913378e1c498913f7cf1ec98913e7878f498913e6838779991be60181f99913661991dd9132661993cdf1f237cf9f85f1e227cf1e00c1c003860c',
                   '00c1c183840830e3e3c7ce1c31f3f3e79f3e31b373641b33799b32641930599b12241830499b12279e3c498b13e79e1cc98b13e78f0e599b13e60387799b13e40183f99b326411a3fd9372661932cdf3e627df3e84f3e227cf1e00400001860c',
                   '00000000000431f1e3e7cf1e31f1e3e7cf1e31913236199379991236199379891a361810498912378e1c49891bf7cf1ec9891be6878f498913e6038779991be60181f999126619b1fd9132661993c5f1f237cf9f85f1e227cf1e840002020604'],
 'chest_TOP': ['000007800000fffc1ff83ff8fffc3ffc3ffe0780703e3c0f0780f00f3c0f0781e00f3c0f0781e0073c0f0781e0073ffe0781e0073ffe0781e0073ffc0781e00f3ff00780f00f3c000780703e3c0007803ffc3c0007801ff83c00000007e00000',
               '000007800000fffc1ff83ff8fffc3ffc3ffe0780703e3c0f0780f00f3c0f0781e00f3c0f0781e0073c0f0781e0073ffe0781e0073ffe0781e0073ffc0781e00f3ff00780f00f3c0007807c3e3c0007803ffc3c0007801ff83c00030007800000',
               '303807e000c0fffc1ff83ff8fffc3ffc3ffe07807c3c3c1f0780f00e3c0f0781e00f3c0f0781e0073c0f0781e0073ffe0781e0073ffe0781e0073ffc0781e00f3ff00780f00f3c0007807c3e3c0007803ffc3c0007801ff83c00000007e00000',
               '787807e007e0fffc1ff83ff87ffc3ffc3ffe07c07c3c3e1f0780f00e3c0f0781e00f3c0f0781e0073c1f0781e0073ffe0781e0073ffc0781e0073ff80781e00f3ff00780f00e3c0007807c3c3c0007803ffc3c0007801ff83c00000007e00000',
               '000007e00000fffc1ff83ff8fffc3ffc3ffe0780703e3c0f0780f00f3c0f0781e00f3c0f0781e0073c0f0781e0073ffe0781e0073ffe0781e0073ffc0781e00f3ff00780f00f3c000780703e3c0007803ffc3c0007801ff83c00030007e01800'],
 'chest_DAH': ['ffc00fe0780ffff00fe0780ff0f80fc0780ff0780fe0780ff03c0ef0780ff03c1ef0780ff03c1c707ffff01c18787ffff01c18787ffff03c3c787ffff03c3ff8780ff03c3ffc780ff0787ffc780ff0f8fffc780ffff0f01e780fffe0e00f780f',
               'ffc00fe0780fffe00fe0780ff0700fc0780ff0380ee0780ff03c0ef0780ff03c1e70780ff01c18707ffff01c18787ffff01c38787ffff03c3ff8780ff03c3ffc780ff0787f7c780ff0f8ff3c780ffff0f01c780fffe0600c780f000000040000',
               'ffc007e0780fffe00fe0780ff0700fc0780ff0381ee0780ff03c1ef0780ff03c1cf0780ff01c18707ffff01c18787ffff01c38787ffff03c3ff8780ff03c3ffc780ff0787f7c780ff0f8ff3c780ffff0f01c780fffc0600c780f1f0000000000',
               '1f0001000000ffc003c0780fffe00fe0780ff1f00fc0780ff03c1ff0780ff03c1cf0780ff03c18787ffff01c18787ffff01c38787ffff01c3c787ffff03c3ff8780ff03c3ff8780ff1f0fffe780fffe0f01e780fffe0e00c380f040000000000',
               'ffc00fe0780ffff80fe0780ff0780fc0780ff0381ee0780ff03c1ef0780ff03c1e70780ff03c18707ffff01c18787ffff01c38787ffff03c3ff8780ff03c3ffc780ff07877fc780ff0f8f39c780ffff0f01e780fffe0600e780f180000002000'],
 'pelvis': 'f001802003f00000fc01802003f80000fc01802003f800008c01800000c00000c471a12380c4253ec6f1b127c0c467bec4d99326c0c4673fcc89932600c46633fc8d922600c46623f9fd9a2380c46423f1f99a23c0c4642381819e20c0c66423c0998e2040c66423c0f98c27c0c7e423c0f98c23c0c3e4238031042380432423',
 'pelvis_Address': ['39f1e3e7ce0e39f1f3e7cf1e39b9f3760d9339993b36199329899a36189129899b16181069899b37ce1869899b37cf1e6d899be7c70f6d899be601837d899b6601917c891b6618b1fc99b32619b344f9f327dd9bc4f1f337cf1ec7e1e337c70e',
                    '31e1e3c7ce0e31f1e3e7cf1e31b13366091371993234199379991a341811599912341810598912378c18498913278f1e498913e7870f798913e6018379991a440181fd9912661191fd99326619938df1f266599b85f1e237cf1e85e1c227c60e',
                    '31e1e3c7ce0e31f1e3e7cf1e31f1f366491379993276199379991236181159991a361810498912378c18498913678f1e498913e7870e799913e60183799b1b440181fd9912641191fdb9b26619938df1f267cd9b8df1e237cf1e85e1c227c70e',
                    '00000000040830e3c3c7ce1c31f3e3e7df1e31f3f3e69b3671bb33661132799b12241830599b12271c30599b12679c3cc98b13678e1e499b13e7870e799b13c40183f99b124411a3f99b326611b2cdf3f6679f368cf3e627df1e85e1c227ce1c',
                    '31e1e3c7cf0e31f1e3e7cf1e31b13266891379993236199379991236181159991a361810599912378c18498912278f1e498913e7870f699913e60183f9991a460191fd99126611b1fd99326619b38df17267dd9b85f1e237cf1f87e1c227c70e'],
 'pelvis_TOP': ['fffc1ff03ff8fffc3ff83ffe0f807c7c3e7e0780f83e3c0f0780f01f3c0f0781e00f3c0f0781e00f3c0f0781e0073c1f0781e0073ffe0781e00f3ffc0781e00f3c000780f01f3c000780f83e3c0007807e7c3c0007803ff83c0007800ff03c00',
                'fffc1ff03ff8fffc3ff83ffe0f807c7c3e1e0780f83e3c0f0780f01f3c0f0781e00f3c0f0781e00f3c0f0781e0073c1e0781e0073ffe0781e00f3ff00781e00f3c000780f01f3c000780f83e3c0007807c7c3c0007803ff83c0007800ff03c00',
                'fffc1ff03ff0fffc3ff83ffe6fd87c7c3e7e0780f83e3c0f0780f01f3c0f0781e00f3c0f0781e00f3c0f0781e0073e1f0781e0073ffe0781e00f3ffc0781e00f3c000780f01f3c000780f83e3c0007807e7c3c0007803ff83c0007800ff03c00',
                '000001000000fffc1ff01ff0fffc3ff83ffe7ff87e7c3f3e0780f03e3c0f0781e00f3c0f0781e00f3c0f0781e00f3e1e0781e0073f3e0781e0073ffc0781e00f3ff80781e00f3e000780f03e3c000780fefc3c0007803ff83c0003800ff03800',
                'fffc1ff03ff0fffc3ff83ffe07807c7c3c7e0780f81e3c0f0780f00f3c0f0781e00f3c0f0781e00f3c0f0781e0073c1e0781e00f3ffe0781e00f3ffc0781e00f3c000780f00f3c000780f81e3c0007807e7c3c0007803ff83c0007800ff03c00'],
 'pelvis_DAH': ['ffc00fe0780fffe00fe0780ff3f00fc0780ff0f80ee0780ff07c0ef0780ff03c1ef0780ff03c1e70780ff03c18787ffff03c38787ffff03c3ef8780ff03c3ff8780ff07c7ffc780ff0f8fffc780ffff0f01c780fffe0e00c780fff81e00e780f',
                'ffc00780780effe00fc0780ff1f00fe0780ff0f80fe0780ff07c0ff0780ff03c1cf0780ff03c1870780ff03c18787ffff03c38787ffff03c3ef8780ff03c3ff8780ff0787ffc780ff0f8fffc780ffbf0f01c780fffe0e00c780fff80e00c780e',
                'ffc00780780effe00fc0780ffbf00fe0780ff0f80fe0780ff07c0ff0780ff03c1cf0780ff03c1870780ff03c18787ffff03c38787ffff03c3ef8780ff03c3ff8780ff0787ffc780ff0f8fffc780ffbf0f01c780fffe0f00c780fff80e00c780e',
                'ff800780380effe00fc0780ffbf00fe0780ff0f81fe0780ff07c1ff0780ff03c1cf0780ff03c18707c1ff01c18787ffff01c38787ffff03c3ef87c1ff03c3ff8780ff0787ff8780ff0f8fff8780ffff0f01c780fffe0e01c780f7f80e00c380e',
                'ffc00fe0780fffe00fe0780ff1f00fe0780ff0f80ec0780ff07c0ee0780ff03c1ef0780ff03c1ef0780ff01c18787ffff03c38787ffff03c3978780ff03c3ff8780ff0787ffc780ff0f8fffc780ffff0f01c780fffe0e00c780fff81e00c780f']}


def _sportsbox_color_mask(crop):
    rgb = np.asarray(crop)
    return (rgb.max(axis=2) > 140) & (np.ptp(rgb, axis=2) > 65)


def _sportsbox_glyphs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask.any(axis=0), False]))
    for left, right in edges.reshape(-1, 2):
        fragment = mask[:, left:right]
        ys = np.flatnonzero(fragment.any(axis=1))
        yield left, right, int(ys[0]), fragment[ys[0]:ys[-1] + 1]


def _sportsbox_fingerprint(mask, size):
    ys, xs = np.where(mask)
    if not xs.size:
        raise ValueError("文字を検出できませんでした。")
    boxed = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sample = Image.fromarray(boxed.astype(np.uint8) * 255).resize(size, SPORTSBOX_RESAMPLE)
    return np.asarray(sample).flatten() >= 128


def _sportsbox_white_mask(crop):
    return np.asarray(crop).min(axis=2) >= 165


def _sportsbox_distance(sample, encoded):
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(encoded), dtype=np.uint8))
    return np.mean(sample != bits)


def _sportsbox_read_number(image, y, bounds, digit_templates):
    left, right = bounds
    cut_off = right > image.width
    scan_left = left + 6
    scan_right = min(right - 6, image.width)
    groups = list(_sportsbox_glyphs(_sportsbox_color_mask(image.crop((scan_left, y, scan_right, y + 62)))))
    if not groups:
        return None
    last_x = groups[-1][1] + scan_left
    has_degree = 12 <= groups[-1][3].shape[0] <= 25 and groups[-1][2] < 20
    if has_degree:
        degree_start = groups[-1][0] + scan_left
        groups.pop()
    if not groups:
        return None
    first_x = groups[0][0] + scan_left
    digit_end = groups[-1][1] + scan_left
    if not has_degree and not cut_off:
        return None
    if cut_off:
        # The reference crop can hide the degree symbol. Infer its expected
        # position from the text alignment, while rejecting clipped digits.
        if has_degree and not 4 <= degree_start - digit_end <= 13:
            return None
        if not has_degree and digit_end + 7 < image.width:
            return None
        if last_x == image.width and not has_degree:
            return None
        if abs((first_x + digit_end + 29) / 2 - (left + right) / 2) > 8:
            return None

    signed = groups[0][3].shape[0] <= 9
    if signed:
        dash = groups.pop(0)[3]
        if not 12 <= dash.shape[1] <= 23 or dash.shape[0] < 4:
            return None
    if not 1 <= len(groups) <= 3:
        return None
    digits = []
    for _, _, _, shape in groups:
        h, w = shape.shape
        if not 35 <= h <= 47 or not 13 <= w <= 36:
            return None
        sample = _sportsbox_fingerprint(shape, (16, 24))
        scores = sorted((min(_sportsbox_distance(sample, ref) for ref in refs), digit) for digit, refs in digit_templates.items())
        if scores[0][0] > .14 or scores[1][0] - scores[0][0] < .035:
            return None
        digits.append(scores[0][1])
    value = int("".join(digits)) * (-1 if signed else 1)
    return value if -180 <= value <= 180 else None


def read_sportsbox_angles(data):
    """Return detected values, leaving unconfirmed cells empty."""
    with Image.open(io.BytesIO(data)) as source:
        if source.width * source.height > 20_000_000:
            raise ValueError("画像が大きすぎます。横長のTurn画面のスクリーンショットを選び直してください。")
        image = ImageOps.exif_transpose(source).convert("RGB")

    width, height = image.size
    aspect = SPORTSBOX_REFERENCE_SIZE[0] / SPORTSBOX_REFERENCE_SIZE[1]
    if min(width, height) < 400:
        raise ValueError("画像が小さすぎます。縮小する前のスクリーンショットを選び直してください。")
    if abs(width / height - aspect) <= .06:
        return _read_sportsbox_layout(image)
    if abs(height / width - aspect) <= .06:
        # Some phones save a landscape capture as a sideways portrait image.
        candidates = (_read_sportsbox_layout(image.transpose(turn))
                      for turn in (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270))
        return max(candidates, key=lambda values: sum(map(len, values.values())))
    raise ValueError("画像は見本と同じ横長のTurn画面にしてください。Chest TurnとPelvis Turnの両方が入る状態で撮影してください。")


def _read_sportsbox_layout(image):
    image = image.resize(SPORTSBOX_REFERENCE_SIZE, SPORTSBOX_RESAMPLE)

    templates = SPORTSBOX_GLYPH_TEMPLATES
    values = {code: {} for code in SPORTSBOX_POSITIONS}
    for metric, (y, cards, title_y) in SPORTSBOX_ROWS.items():
        for (left, right), code, badge_box, badge in zip(cards, SPORTSBOX_POSITIONS, SPORTSBOX_BADGES[metric], ("Address", "TOP", "DAH")):
            title_box = (left + 6, title_y, left + 145, title_y + 29)
            try:
                title = _sportsbox_fingerprint(_sportsbox_white_mask(image.crop(title_box)), (64, 16))
                position = _sportsbox_fingerprint(_sportsbox_white_mask(image.crop(badge_box)), (48, 16))
            except ValueError:
                continue
            if _sportsbox_distance(title, templates[metric]) > .15:
                continue
            if min(_sportsbox_distance(position, ref) for ref in templates[metric + "_" + badge]) > .14:
                continue
            value = _sportsbox_read_number(image, y, (left, right), templates["digits"])
            if value is not None:
                values[code][metric] = value
    return values


def clear_sportsbox_image() -> None:
    job = st.session_state.get("job")
    if job:
        job.pop("sportsbox_import", None)
    clear_outputs()


def sportsbox_settings(job: dict) -> None:
    token = job["token"]
    job["sportsbox_angles"] = {}
    if not st.checkbox("Sportsboxの回転角を表示する（オプション）", key=f"sportsbox_enabled_{token}", on_change=clear_outputs):
        return

    user_agent = st.context.headers.get("User-Agent", "")
    default_mobile = bool(re.search(r"Android|iPhone|iPad|iPod|Mobile", user_agent, re.I))
    methods = ("通常", "スマホ向け（分割送信）", "手入力")
    method = st.radio(
        "Sportsboxの入力方法", methods, index=int(default_mobile),
        key=f"sportsbox_method_{token}", horizontal=True, on_change=clear_outputs,
    )
    if method == "スマホ向け（分割送信）":
        transport = job.setdefault("sportsbox_upload", {})
        screenshot = mobile_upload("image", transport, f"sportsbox_mobile_{token}")
    elif method == "通常":
        screenshot = st.file_uploader(
            "同じスイングのSportsbox画像を選択",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"sportsbox_file_{token}", on_change=clear_sportsbox_image,
            help="見本と同じ横長のTurn画面を使用してください。DAHの次のカードをIMPACTとして読み取ります。",
        )
    else:
        screenshot = None

    state = job.get("sportsbox_import")
    if method == "手入力":
        source_id = "manual"
    elif screenshot is not None:
        source_id = screenshot.file_id
        st.caption(f"画像を受信しました　{screenshot.name}")
    elif state and state.get("method") == method:
        source_id = state["id"]
        st.caption(f"保存された読み取り結果　{state.get('name', '')}")
    else:
        return

    if state is None or state["id"] != source_id:
        clear_outputs()
        error = None
        values = {}
        if screenshot is not None:
            with st.spinner("Sportsbox画像から回転角を読み取っています…"):
                try:
                    values = read_sportsbox_angles(screenshot.getbuffer())
                except ValueError as exc:
                    error = str(exc)
                except (OSError, Image.DecompressionBombError):
                    error = "画像を開けませんでした。端末のスクリーンショットをPNGまたはJPEGで選び直してください。"
        state = {
            "id": source_id, "method": method, "name": screenshot.name if screenshot is not None else "",
            "values": values, "error": error, "read_count": sum(map(len, values.values())),
        }
        job["sportsbox_import"] = state

    if state["error"]:
        st.warning(state["error"])
    elif method != "手入力":
        count = state["read_count"]
        if count:
            st.caption(f"画像から読み取った値　{count} / 6")
        else:
            st.warning("画像は受信できましたが、回転角を読み取れませんでした。見本と同じTurn画面を選ぶか、下の欄に入力してください。")
    st.caption("画像と数値を照合し、必要に応じて修正してください。両方の角度がそろったポジションに表示します。")
    values = {}
    for col, (code, position) in zip(st.columns(3), SPORTSBOX_POSITIONS.items()):
        col.markdown(f"**{code} · {position}**")
        values[code] = {}
        for metric, name in SPORTSBOX_METRICS.items():
            values[code][metric] = col.number_input(
                f"{name}の回転角（°）", min_value=-180, max_value=180,
                value=state["values"].get(code, {}).get(metric), step=1, format="%d",
                placeholder="未読取",
                key=f"sportsbox_{token}_{source_id}_{code}_{metric}", on_change=clear_outputs,
            )
    state["values"] = values
    job["sportsbox_angles"] = values
    if any(value is None for pair in values.values() for value in pair.values()):
        st.info("読み取れなかった値は空欄です。画像を見ながら入力できます。")


WORKSPACE_HTML = """<!doctype html><html lang="ja"><meta charset="utf-8"><script>
const post=(type, extra={})=>window.parent.postMessage({isStreamlitMessage:true,type,...extra},'*');
let reported=false;
window.addEventListener('message',event=>{
    if(reported||event.source!==window.parent||event.data.type!=='streamlit:render')return;
    reported=true;
    let id=null;
    try{
        const key='golf-swing-edit-workspace';
        id=sessionStorage.getItem(key);
        if(!/^[0-9a-f]{32}$/.test(id||'')){
            id=Array.from(crypto.getRandomValues(new Uint8Array(16)),byte=>byte.toString(16).padStart(2,'0')).join('');
            sessionStorage.setItem(key,id);
        }
    }catch{id=null;}
    post('streamlit:setComponentValue',{dataType:'json',value:{id}});
});
post('streamlit:componentReady',{apiVersion:1});
post('streamlit:setFrameHeight',{height:0});
</script></html>"""

WORKSPACE_SETTINGS = {
    "active_p", "active_line", "camera_mode", "enabled_lines", "styles",
    "show_p1_reference", "target_side", "custom_positions", "show_phase_list",
    "pause_positions", "pause_seconds", "snapshot_positions", "upload_method",
    "back_vertical_dash_initialized", "swing_handedness",
}


def workspace_file(workspace_id: str) -> Path:
    return Path(tempfile.gettempdir()) / "golf_swing_workspace" / f"{workspace_id}.json"


def restore_workspace() -> None:
    folder = Path(tempfile.gettempdir()) / ("golf_workspace_ui_" + hashlib.sha256(WORKSPACE_HTML.encode()).hexdigest()[:16])
    folder.mkdir(exist_ok=True)
    index = folder / "index.html"
    if not index.exists():
        index.write_text(WORKSPACE_HTML, encoding="utf-8")
    component = declare_component("golf_workspace", path=str(folder))
    event = component(key="workspace_browser", default=None)
    state = st.session_state
    if "workspace_id" in state or not isinstance(event, dict):
        return
    workspace_id = event.get("id")
    if not isinstance(workspace_id, str) or not re.fullmatch(r"[0-9a-f]{32}", workspace_id):
        state["workspace_id"] = None
        return
    state["workspace_id"] = workspace_id
    if state.get("job"):
        return
    try:
        record = json.loads(workspace_file(workspace_id).read_text(encoding="utf-8"))
        job = record["job"]
        settings = record["settings"]
        if not isinstance(settings, dict):
            raise ValueError("Invalid workspace settings")
        if not Path(job["path"]).is_file():
            state["workspace_unavailable"] = True
            return
    except FileNotFoundError:
        return
    except (OSError, ValueError, KeyError, TypeError):
        state["workspace_unavailable"] = True
        return
    job["output_images"] = None
    if job.get("output_video") and not Path(job["output_video"]).is_file():
        job["output_video"] = None
    state["job"] = job
    for key, value in settings.items():
        state[key] = value
    state["restore_view"] = uuid.uuid4().hex


def save_workspace() -> None:
    state = st.session_state
    workspace_id, job = state.get("workspace_id"), state.get("job")
    if not workspace_id or not job:
        return
    token = job["token"]
    keys = WORKSPACE_SETTINGS | {f"sportsbox_enabled_{token}", f"sportsbox_method_{token}"}
    settings = {key: state[key] for key in keys if key in state}
    saved_job = {key: value for key, value in job.items() if key not in {"sportsbox_upload", "output_images"}}
    payload = json.dumps({"job": saved_job, "settings": settings}, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    if digest == state.get("workspace_digest"):
        return
    path = workspace_file(workspace_id)
    temporary = None
    try:
        path.parent.mkdir(mode=0o700, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        temporary.replace(path)
        state["workspace_digest"] = digest
        state.pop("workspace_save_failed", None)
    except OSError:
        state["workspace_save_failed"] = True
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def main():
    st.set_page_config(page_title="ゴルフスイング分析・簡単操作版",page_icon="🏌️",layout="wide",initial_sidebar_state="expanded")
    st.markdown("""<style>
    html, body, [data-testid="stMain"] {overscroll-behavior: none;}
    </style>""", unsafe_allow_html=True)
    restore_workspace()
    try:
        render_app()
    finally:
        save_workspace()


def render_app():
    st.title("🏌️ ゴルフスイング分析")
    st.caption("Python / Streamlit版・簡単操作版　｜　動画選択 → 自動解析 → コマ・線を調整 → 保存")
    st.caption("撮影アングル・線の表示設定は左側の「管理線の設定」にあります。見えない場合は画面左上の「＞」または「≫」で開いてください。")
    if "camera_mode" not in st.session_state:st.session_state["camera_mode"]="back"
    styles = st.session_state.setdefault("styles", {})
    rear_mode = st.session_state["camera_mode"] == "back"
    st.session_state.setdefault("enabled_lines", (DEFAULT_LINES if rear_mode else FRONT_DEFAULT_LINES)[:])
    if styles and "back_vertical" not in styles and rear_mode:
        current_lines = st.session_state["enabled_lines"]
        st.session_state["enabled_lines"] = list(dict.fromkeys([*current_lines, *REAR_ONLY_LINES]))
    if styles and "right_leg_outer" not in styles and not rear_mode:
        current_lines = st.session_state["enabled_lines"]
        st.session_state["enabled_lines"] = list(dict.fromkeys([*current_lines, *FRONT_ONLY_LINES]))
    for key, color in COLORS.items():
        styles.setdefault(key, {"color":color, "width":3, "dash":"dashed" if key in ("hip", "back_vertical") else "solid", "positions":[]})
    if not st.session_state.get("back_vertical_dash_initialized"):
        vertical_style = styles["back_vertical"]
        if vertical_style["color"].upper() == COLORS["back_vertical"] and vertical_style["dash"] == "solid":
            vertical_style["dash"] = "dashed"
            st.session_state.pop("dash_back_vertical", None)
        st.session_state["back_vertical_dash_initialized"] = True
    if styles["shoulders"]["color"].upper() == "#69F0AE":
        styles["shoulders"]["color"] = COLORS["shoulders"]
        st.session_state.pop("color_shoulders", None)
    with st.sidebar:
        st.subheader("管理線の設定")
        a,b=st.columns(2)
        a.button("後方撮影",on_click=switch_mode,args=(False,),use_container_width=True)
        b.button("正面撮影",on_click=switch_mode,args=(True,),use_container_width=True)
        unavailable = FRONT_ONLY_LINES if rear_mode else REAR_ONLY_LINES
        available = [key for key in LINE_NAMES if key not in unavailable]
        enabled=st.multiselect("表示する線",available,format_func=lambda k:LINE_NAMES[k],key="enabled_lines",on_change=clear_outputs)
        st.caption("線はプレビュー上の丸、またはシャフト・ヒップの線をドラッグして動かせます。")
        if st.session_state["camera_mode"]=="back":
            st.checkbox("P1の胸・腰の基準線（白点線・太さ2）",value=True,key="show_p1_reference",on_change=clear_outputs)
        if "tilt" in enabled:
            st.selectbox("目標方向（画面上）",["左","右"],key="target_side",on_change=clear_outputs,help="胸が腰より目標側なら正、反対側ならマイナスです。")
        custom_positions=st.multiselect("共通の手描き図形を表示するP",P_CODES,default=[],key="custom_positions",help="未選択なら全フレームに表示します。",on_change=clear_outputs)
        with st.expander("読み込みの詳細設定"):
            do_ai=st.checkbox("AI骨格解析を行う",value=True)
            max_side=st.selectbox("解析用動画の長辺",[720,960],index=1,format_func=lambda x:f"{x}px")
            st.caption("元動画のフレームレートを最大60fpsまで保持し、選択した解像度に変換します。")
    st.subheader("1. 動画を選ぶ")
    handedness = st.radio("スイングの利き打ち", ["右打ち", "左打ち"], horizontal=True, key="swing_handedness")
    upload_method=st.radio("動画の取り込み方法",["通常（PCなど）","Android向け（分割送信）"],horizontal=True,key="upload_method")
    if upload_method=="Android向け（分割送信）":
        uploaded=mobile_upload("video", st.session_state.setdefault("android_upload_state", {}), "android_upload_widget")
    else:
        uploaded=st.file_uploader("動画を選択",type=None,help="ファイル名が出ない場合は、上の「Android向け（分割送信）」を選んでください。")
    if uploaded is not None:
        st.caption(f"受信済み：{uploaded.name}　{uploaded.size/1024/1024:.1f} MB")
    if st.button("動画を読み込んで解析",type="primary",disabled=uploaded is None):
        folder=Path(tempfile.mkdtemp(prefix="golf_simple_"));progress=st.progress(0,text="動画を受信しました。変換しています…")
        try:
            suffix=Path(uploaded.name).suffix.lower()
            if not re.fullmatch(r"\.[a-z0-9]{1,8}",suffix):suffix=".mp4"
            source=folder/("source"+suffix)
            with source.open("wb") as f:f.write(uploaded.getbuffer())
            path=folder/"working.mp4";info=normalize_video(source,path,max_side)
            if info["seconds"]>60:raise ValueError("1回のスイングを60秒以内に切り出してから読み込んでください。")
            poses=[None]*info["frames"]
            if do_ai:
                try:
                    progress.progress(0,text="AIモデルを準備しています（初回はダウンロードが必要です）…")
                    poses=analyze_skeleton(path,info,lambda v,t:progress.progress(v,text=t))
                    poses=poses[:info["frames"]]+[None]*max(0,info["frames"]-len(poses))
                except Exception as exc:
                    st.warning(f"骨格解析ができませんでした：{exc}\nシャフト・ヒップと手描き線は利用できます。")
            previous=st.session_state.get("job")
            st.session_state["job"]=build_job(path, info, poses, folder, uploaded.name, left_handed=handedness == "左打ち")
            st.session_state["active_p"]="P1"
            st.session_state.pop("restore_view", None)
            if previous:shutil.rmtree(previous["folder"],ignore_errors=True)
            progress.empty()
            st.success(f"読み込み完了　{sum(p is not None for p in poses)}/{len(poses)}コマで骨格を検出")
        except Exception as exc:
            progress.empty();shutil.rmtree(folder,ignore_errors=True);st.error(str(exc))
    job=st.session_state.get("job")
    if not job:
        if st.session_state.get("workspace_unavailable"):
            st.info("前回の編集データを復元できませんでした。動画を選び直してください。")
        st.info("動画を選んで「動画を読み込んで解析」を押してください。")
        return
    sportsbox_settings(job)
    job["camera_mode"] = st.session_state["camera_mode"]
    if "back_vertical" not in job["guides"]:
        reference = get_pose(job, job["p_frames"]["P1"], "P1")
        job["guides"]["back_vertical"] = rear_reference_x(reference)
    job["show_p1_spine"] = rear_mode and st.session_state.get("show_p1_reference", True)
    job["target_dir"]=-1 if st.session_state.get("target_side","左")=="左" else 1
    info=job["info"]
    st.divider();st.subheader("2. コマと線を調整する")
    if st.session_state.get("restore_view"):
        st.caption("前回の編集内容を復元しました。")
    if st.session_state.get("workspace_save_failed"):
        st.warning("編集内容を一時保存できませんでした。ページを閉じずに作業してください。")
    code=st.session_state.get("active_p","P1")
    if code not in P_CODES:code="P1"
    c2,color_col,width_col=st.columns([2.4,1,1.2])
    enabled=st.session_state["enabled_lines"]
    line=c2.selectbox("調整する線",enabled or ["shaft"],format_func=lambda k:LINE_NAMES[k],index=enabled.index("shaft") if "shaft" in enabled else 0,key="active_line")
    style=st.session_state["styles"][line]
    style["color"]=color_col.color_picker("線の色",style["color"],key=f"color_{line}",on_change=clear_outputs)
    style["width"]=width_col.selectbox("線の太さ",[1,2,3,4,5,6,8],index=[1,2,3,4,5,6,8].index(style["width"]),format_func=lambda v:f"{v}px",key=f"width_{line}",on_change=clear_outputs)
    if line=="head":st.caption("点線の円はP1の基準位置、もう1つの円は頭の動きに追従します。丸をドラッグすると基準位置・大きさを調整できます。")
    if line=="tilt":st.caption("白い点線が垂直＝0°です。胸が腰より目標側なら正、反対側ならマイナスで表示します。目標方向は左側の管理線設定で変更できます。")
    if line == "back_vertical":
        st.caption("P1の頭頂から足元までを目安にした垂直線です。丸をドラッグすると左右に移動します。")
    for line_code, points, part in (
        ("hip_right_knee", ("rk",), "右ひざ"),
        ("right_leg_outer", RIGHT_LEG_POINTS, "右脚の外側の位置"),
    ):
        if line_code not in enabled:
            continue
        detected = (right_leg_reference(job) is not None if line_code == "right_leg_outer"
                    else any(pose and valid_point(pose.get("rk")) for pose in job["poses"]))
        if not detected:
            if st.button(f"編集中の動画で{part}を検出"):
                progress = st.progress(0, text=f"{part}を検出しています…")
                try:
                    found = refresh_pose_points(job, points, lambda value, text: progress.progress(value, text=text))
                    if line_code == "right_leg_outer" and right_leg_reference(job) is None:
                        st.info("P1で右脚の外側を検出できませんでした。P1のコマを前後に調整してください。")
                    elif found:
                        st.success(f"{found}コマで{part}を検出しました。")
                    else:
                        st.info(f"{part}を検出できませんでした。")
                except Exception as exc:
                    st.error(f"{part}を検出できませんでした。{exc}")
                finally:
                    progress.empty()
    if line == "hip_right_knee":
        st.caption("右ひざを検出したコマに、画面の水平線（0°）に対する角度を表示します。腰側とひざ側の丸をドラッグして調整できます。")
    if line == "right_leg_outer":
        st.caption("P1の外くるぶしと太ももの付け根を結び、同じ位置・角度でP4まで表示します。P1で両端の丸をドラッグして調整できます。角度の基準は画面の水平線（0°）です。")
    with st.expander("点線・表示するPを変更"):
        style["dash"]=st.selectbox("種類",["solid","dashed"],index=["solid","dashed"].index(style["dash"]),format_func=lambda x:"実線" if x=="solid" else "点線",key=f"dash_{line}",on_change=clear_outputs)
        if line not in ("shoulders", "right_leg_outer"):
            style["positions"]=st.multiselect("この線を表示するポジション（複数選択可）",P_CODES,default=style["positions"],key=f"positions_{line}",help="未選択なら全フレームに表示します。",on_change=clear_outputs)
    if "shoulders" in enabled:
        shoulder_style=st.session_state["styles"]["shoulders"]
        shoulder_style["positions"]=st.multiselect("両肩ラインを表示するP（複数選択）",P_CODES,default=shoulder_style["positions"],key="positions_shoulders",help="未選択なら全フレーム。選んだ場合は、そのPに設定したコマだけに表示します。",on_change=clear_outputs)
    cfg={key:{**copy.deepcopy(value),"enabled":key in enabled} for key,value in st.session_state["styles"].items()}
    widget=f"frame_{job['token']}_{code}"
    if widget not in st.session_state:st.session_state[widget]=job["p_frames"][code]
    st.markdown("""<style>
    .st-key-compact_preview, .st-key-compact_preview [data-testid="stVerticalBlock"] {gap: 0.25rem;}
    .st-key-compact_preview [data-testid="stHorizontalBlock"] {gap: 0.5rem;}
    .st-key-compact_preview [data-testid="stSlider"] {padding-top: 0; padding-bottom: 0;}
    </style>""",unsafe_allow_html=True)
    with st.container(key="compact_preview"):
        prev_col,slider_col,next_col=st.columns([1.5,6,1.5],vertical_alignment="center")
        prev_col.button("◀ 前",help="1コマ前",on_click=step_frame,args=(code,widget,-1),use_container_width=True)
        slider_col.slider("コマ番号",0,info["frames"]-1,key=widget,on_change=frame_change,args=(code,widget),label_visibility="collapsed")
        next_col.button("次 ▶",help="1コマ後",on_click=step_frame,args=(code,widget,1),use_container_width=True)
        frame_index=int(st.session_state[widget]);job["p_frames"][code]=frame_index
        current=read_frame(job["path"],frame_index)
        ok,jpg=cv2.imencode(".jpg",current,[cv2.IMWRITE_JPEG_QUALITY,88])
        body=get_pose(job,frame_index,code)
        if code == "P1" and (reference := right_leg_reference(job)):
            body = {**(body or {}), **dict(zip(RIGHT_LEG_POINTS, reference))}
        component=editor_component()
        event=component(image=base64.b64encode(jpg).decode(),width=info["width"],height=info["height"],scene=f"{job['token']}:{code}:{frame_index}",phase=code,frame=frame_index,guides=job["guides"],body=body,primitives=primitives_for({**job,"shapes":[]},frame_index,cfg,code,custom_positions),shapes=job["shapes"],active_line=line,custom_visible=in_positions(frame_index,custom_positions,job["p_frames"]),sportsbox=sportsbox_overlay(job,code),restore_view=st.session_state.get("restore_view"),ack=job["last_event"],key=f"editor_{job['token']}",default=None)
        next_phase=editor_phase_change(job,event,code,frame_index)
        if next_phase:
            st.session_state["active_p"]=next_phase
            st.rerun()
        if accept_editor_event(job,event,code,frame_index):st.rerun()
    st.caption(f"{code}　｜　{frame_index}コマ / {frame_index/info['fps']:.2f}秒　｜　コマ調整・線の移動は自動反映")
    if not body:st.caption("このコマでは骨格を検出できませんでした。別のコマに調整するか、手描き直線を使ってください。")
    if job.get("position_status") == "manual":
        st.warning("スイングの流れを判定できなかったため、コマは等間隔の仮置きです。映像を見ながら各Pを設定してください。")
    elif job.get("position_status") == "review":
        st.warning("腕の水平やフィニッシュを判定しにくい箇所があります。P一覧で候補を確認してください。")
    st.caption("P2・P6・P8は手元の高さ、P7は手元が下がってから上がる動きで推定します。シャフトの水平とボールの接触は映像で確認してください。")
    if st.checkbox("P1〜P10を一覧表示",key="show_phase_list"):
        for start in range(0,10,5):
            cols=st.columns(5)
            for col,p in zip(cols,P_CODES[start:start+5]):
                idx=job["p_frames"][p]
                col.image(cv2.cvtColor(draw_annotations(read_frame(job["path"],idx),primitives_for(job,idx,cfg,p,custom_positions),p,sportsbox_overlay(job,p)),cv2.COLOR_BGR2RGB),caption=f"{p} {idx}コマ",use_container_width=True)
    st.divider();st.subheader("3. 作成して保存する")
    a,b=st.columns(2)
    pauses=a.multiselect("動画で止めるP",P_CODES,default=["P1","P4","P7"],key="pause_positions",on_change=clear_outputs)
    snapshots=b.multiselect("静止画にするP",P_CODES,default=["P1","P4","P7"],key="snapshot_positions",on_change=clear_outputs)
    seconds=st.slider("動画での停止時間（秒）",.5,5.,2.,.5,key="pause_seconds",on_change=clear_outputs)
    st.caption("線を表示するPとは別に、上の「動画で止めるP」で停止対象を選びます。変更後は分析動画を作り直してください。")
    if pauses:
        plan=pause_plan(job,pauses,seconds)
        st.caption("停止予定："+" ／ ".join(f"{p}：{job['p_frames'][p]}コマ（元動画 {job['p_frames'][p]/info['fps']:.2f}秒）" for p in pauses))
        if any(len(codes)>1 for codes in plan.values()):
            st.info("同じコマに設定されたPがあります。各Pの名前を表示して順番に停止します。別の瞬間で止める場合は、上のコマ調整で位置を分けてください。")
    else:
        st.info("停止するPが未選択です。ポーズさせる場合は「動画で止めるP」を選んでください。")
    a,b=st.columns(2)
    if a.button("🎬 分析動画を作成",type="primary",use_container_width=True):
        progress=st.progress(0,text="動画を作成しています…")
        try:
            job["output_video"]=str(export_video(job,cfg,pauses,seconds,custom_positions,lambda v,t:progress.progress(v,text=t)))
            st.success("動画を作成しました。下のボタンで保存してください。")
        except Exception as exc:st.error(str(exc))
        finally:progress.empty()
    if b.button("📸 選択した静止画を作成",use_container_width=True):
        try:
            with st.spinner("静止画を作成しています…"):job["output_images"]=export_images(job,cfg,snapshots,custom_positions)
        except Exception as exc:st.error(str(exc))
    name=re.sub(r"[^\w\-]","_",Path(job["name"]).stem)[:80] or "swing"
    if job.get("output_video"):
        data=Path(job["output_video"]).read_bytes()
        st.video(data)
        st.download_button("📥 分析動画を保存（MP4）",data,name+"_analyzed.mp4","video/mp4",on_click="ignore",use_container_width=True)
        st.caption("分析動画は音声なしです。スマホで保存する場合はダウンロードしたファイルを確認してください。")
    if job.get("output_images"):
        data=job["output_images"]
        a,b=st.columns(2)
        a.download_button("📥 静止画一式を保存（ZIP）",data["zip"],name+"_snapshots.zip","application/zip",on_click="ignore",use_container_width=True)
        b.download_button("📥 連続写真を保存（PNG）",data["sheet"],name+"_sequence.png","image/png",on_click="ignore",use_container_width=True)
        st.image(data["sheet"],caption="連続写真",use_container_width=True)
        with st.expander("個別の静止画を保存"):
            for p,png in data["images"]:st.download_button(p+"を保存",png,name+"_"+p+".png","image/png",key=f"download_{p}",on_click="ignore")


if __name__ == "__main__":
    main()
