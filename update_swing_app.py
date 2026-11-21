from pathlib import Path

folder = Path(__file__).resolve().parent
source = folder / "app(8).py"
output = folder / "app_updated.py"
code = source.read_text(encoding="utf-8-sig")


def replace(old, new):
    global code
    if code.count(old) != 1:
        raise SystemExit(f"現行コードと一致しない箇所があります。\n{old}")
    code = code.replace(old, new, 1)


replace(
    'FRONT_ONLY_LINES = ("right_leg_outer",)',
    '''LINE_NAMES["pelvis_circle"] = "骨盤中心の左右移動（円）"
COLORS["pelvis_circle"] = "#FF9500"
FRONT_ONLY_LINES = ("right_leg_outer", "pelvis_circle")''',
)

replace(
    '    return pose\n',
    '''    pose["pelvis_visible"] = all(
        float(lm[i].visibility) >= .55 for i in (23, 24)
    ) and valid_point(pose["hip"])
    return pose


def shoulder_heading(image_landmarks, world_landmarks):
    """左右の肩を結ぶ方向の水平面上の角度を返す。"""
    if len(world_landmarks) <= 12:
        return None
    if not all(float(image_landmarks[i].visibility) >= .60 for i in (11, 12)):
        return None
    left, right = world_landmarks[11], world_landmarks[12]
    dx, dz = float(left.x - right.x), float(left.z - right.z)
    if not all(map(math.isfinite, (dx, dz))) or math.hypot(dx, dz) < .10:
        return None
    return math.atan2(dz, dx)


def shoulder_turn(job, phase):
    angles = []
    for code in ("P1", phase):
        index = job["p_frames"][code]
        pose = job["poses"][index] if 0 <= index < len(job["poses"]) else None
        heading = (pose or {}).get("shoulder_heading")
        if heading is None:
            return None
        angles.append(heading)
    delta = angles[1] - angles[0]
    signed = math.atan2(math.sin(delta), math.cos(delta))
    return round(math.degrees(signed) * job.get("target_dir", -1))
''',
)

replace(
    '                poses.append(pose_to_dict(result.pose_landmarks[0], info["width"], info["height"]) if result.pose_landmarks else None)',
    '''                pose = None
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    pose = pose_to_dict(landmarks, info["width"], info["height"])
                    if result.pose_world_landmarks:
                        pose["shoulder_heading"] = shoulder_heading(
                            landmarks, result.pose_world_landmarks[0]
                        )
                poses.append(pose)''',
)

replace(
    '        elif code in ("spine","tilt") and body:',
    '''        elif code == "pelvis_circle":
            p1 = job["p_frames"]["P1"]
            poses = job["poses"]
            ref = (poses[p1] or {}) if 0 <= p1 < len(poses) else {}
            current = (poses[frame] or {}) if 0 <= frame < len(poses) else {}
            origin = ref.get("hip") if ref.get("pelvis_visible") else None
            anchor = g.get("pelvis_center") or origin
            if anchor:
                radius = g.get("pelvis_radius", .045)
                base = {**s, "type": "circle", "radius": radius, "anchor": anchor}
                out.append({**base, "id": "pelvis_reference", "center": anchor,
                            "dash": "dashed", "dx": 0})
                if origin and current.get("pelvis_visible"):
                    dx = current["hip"][0] - origin[0]
                    out.append({**base, "id": "pelvis_circle", "dash": "solid",
                                "center": [anchor[0] + dx, anchor[1]], "dx": dx})
        elif code in ("spine","tilt") and body:''',
)

replace(
    '''        elif code == "hip_right_knee" and body and valid_point(body.get("rk")):
            add_line(code, body["hip"], body["rk"], angle=True)''',
    '''        elif code == "hip_right_knee":
            if frame == job["p_frames"]["P1"] and phase in (None, "P1"):
                knee_pose = job["body_overrides"].get("P1") or job["poses"][frame]
                if knee_pose and all(valid_point(knee_pose.get(k)) for k in ("hip", "rk")):
                    add_line(code, knee_pose["hip"], knee_pose["rk"], angle=True)''',
)
replace(
    'code != "right_leg_outer" and not in_positions',
    'code not in ("right_leg_outer", "hip_right_knee") and not in_positions',
)

replace(
    '        if valid_point(g.get("head")):',
    '''        if phase == "P1" and valid_point(g.get("pelvis_center")):
            job["guides"]["pelvis_center"] = g["pelvis_center"]
            radius = g.get("pelvis_radius")
            if isinstance(radius, (int, float)) and .01 <= radius <= .25:
                job["guides"]["pelvis_radius"] = radius
            changed = True
        if valid_point(g.get("head")):''',
)
replace(
    '    if code=="P1" and not job.get("guides_manual"):',
    '''    if code == "P1":
        job["guides"].pop("pelvis_center", None)
    if code=="P1" and not job.get("guides_manual"):''',
)

replace(
    "if(s.id==='head_reference'){",
    """if(s.id==='pelvis_reference'||s.id==='pelvis_circle'){
const p=g.pelvis_center||s.anchor;
s.center=[p[0]+s.dx,p[1]];
s.radius=g.pelvis_radius||s.radius;
}if(s.id==='head_reference'){""",
)
replace(
    "if(args.active_line==='head'&&",
    """if(args.active_line==='pelvis_circle'&&args.phase==='P1'){
const raw=args.primitives.find(s=>s.id==='pelvis_reference');
if(raw){const s=geometry(raw);
add(s.center,'pelvisCenter');
add([s.center[0]+s.radius,s.center[1]],'pelvisRadius',null,{center:s.center});}
}if(args.active_line==='head'&&""",
)
replace(
    "}else if(d.type==='head'){",
    """}else if(d.type==='pelvisCenter'){
g.pelvis_center=p;changedG=true;
}else if(d.type==='pelvisRadius'){
g.pelvis_center=d.center;
g.pelvis_radius=clamp(Math.hypot((p[0]-d.center[0])*canvas.width,
(p[1]-d.center[1])*canvas.height)/canvas.width,.01,.25);
changedG=true;
}else if(d.type==='head'){""",
)

replace(
    '    panels = []',
    '''    panels = []
    if job.get("camera_mode") == "front" and job.get("show_shoulder_turn", True):
        angle = shoulder_turn(job, phase)
        if angle is not None:
            value = f"{angle:+d}°" if angle else "0°"
            panels.append([[("肩の回転（推定）  " + value, "white")]])''',
)
replace(
    '    "back_vertical_dash_initialized", "swing_handedness",',
    '    "back_vertical_dash_initialized", "swing_handedness", "show_shoulder_turn",',
)
replace(
    '        if "tilt" in enabled:',
    '''        if not rear_mode:
            st.checkbox("肩の回転（推定）を表示", value=True,
                        key="show_shoulder_turn", on_change=clear_outputs)
        if "tilt" in enabled or not rear_mode:''',
)
replace(
    '    job["camera_mode"] = st.session_state["camera_mode"]',
    '''    job["camera_mode"] = st.session_state["camera_mode"]
    job["show_shoulder_turn"] = st.session_state.get("show_shoulder_turn", True)
    if not rear_mode and not any(
        pose and "pelvis_visible" in pose for pose in job["poses"]
    ):
        st.caption("骨盤の円と肩の回転を使うには、この動画を読み込み直してAI骨格解析を行ってください。")''',
)
replace(
    '        if line not in ("shoulders", "right_leg_outer"):',
    '        if line not in ("shoulders", "right_leg_outer", "hip_right_knee"):',
)
replace(
    '右ひざを検出したコマに、画面の水平線（0°）に対する角度を表示します。腰側とひざ側の丸をドラッグして調整できます。',
    '後方撮影のP1だけに表示します。P1で腰側とひざ側の丸をドラッグして調整できます。',
)

compile(code, str(output), "exec")
with output.open("x", encoding="utf-8") as stream:
    stream.write(code)
print(f"作成しました  {output}")