"""3D MuJoCo visualisation of a fruit fly "smoking" a cigarette and
"scrolling reels" on a phone, built on the TuragaLab ``flybody`` fruit-fly
model (flybody/flybody/fruitfly/assets/fruitfly.xml).

This is a playful computational-neuroscience toy visualisation, not a
physically-simulated agent: the fly's pose is written directly into
``qpos`` every frame and ``mujoco.mj_forward`` is called to update
kinematics -- there is no physics stepping (no ``mj_step``).

--------------------------------------------------------------------------
Model scale facts, measured directly from the flybody asset (see the
docstring of ``_measure_body_length`` and the exploration notes in the
project's task report):

  * The model uses CGS units: ``option gravity="0 0 -981"`` cm/s^2, so
    1 MuJoCo unit == 1 cm.
  * flybody's own spawn height (``flybody.fruitfly.fruitfly._SPAWN_POS``)
    is z = 0.1278 (cm), i.e. thorax height above the floor when the legs
    are in their rest/standing configuration.
  * At the model's rest joint configuration (all leg qpos == 0), the legs
    are *already* a plausible standing stance (not a T-pose) -- the mesh's
    zero pose was rigged that way. Wings, however, rest flat out to the
    sides at qpos == 0 and need folding, which we do by copying
    ``model.qpos_spring`` into the wing joints (the same trick flybody
    itself uses to retract unused wings).
  * Body length (rostrum tip to abdomen tip) at that rest pose is
    ~0.25 model units == ~2.5 mm, matching a real fruit fly.

We use BODY_LENGTH = 0.25 (cm) as the "1 fly body length" unit for sizing
props and the arena.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco
from dm_control import mjcf
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from fly3d import reels_media, reels_ui

# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
FLY_XML = os.path.join(_PROJECT_ROOT, "flybody", "flybody", "fruitfly",
                        "assets", "fruitfly.xml")
REELS_ASSETS_DIR = os.path.join(_PROJECT_ROOT, "renders", "assets", "reels")

# ---- measured model scale facts (see module docstring) ----
BODY_LENGTH = 0.25          # cm; nose (rostrum) to abdomen tip at rest pose
SPAWN_Z = 0.1278            # cm; thorax height above floor, standing pose
HEAD_HEIGHT = SPAWN_Z + 0.045   # approx head height above floor, standing

# world arena: env's [0,1]^2 square spans this many fly body-lengths
ARENA_BODY_LENGTHS = 22.0
ARENA_SCALE = ARENA_BODY_LENGTHS * BODY_LENGTH   # cm per 1.0 env unit
FLOOR_Z = 0.0

RENDER_W, RENDER_H = 1280, 720

# --------------------------------------------------------------------------
# Small numpy quaternion / rotation helpers (MuJoCo quat order: w, x, y, z)
# --------------------------------------------------------------------------


def quat_wxyz_from_rotation(rot: Rotation) -> np.ndarray:
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def quat_from_yaw(yaw: float) -> np.ndarray:
    return quat_wxyz_from_rotation(Rotation.from_euler("z", yaw))


def quat_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    """Yaw about world z, then pitch about the new local y (nose up/down)."""
    return quat_wxyz_from_rotation(
        Rotation.from_euler("zy", [yaw, pitch]))


def look_at_quat(cam_pos: np.ndarray, target: np.ndarray,
                  world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion for a camera at cam_pos looking at target.

    MuJoCo camera convention: the camera looks along its local -Z axis,
    with local +Y as up and local +X as right.
    """
    forward = np.asarray(target, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64)
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        forward = np.array([1.0, 0.0, 0.0])
    else:
        forward = forward / norm
    up = np.asarray(world_up, dtype=np.float64)
    if abs(np.dot(forward, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    true_up = np.cross(right, forward)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-12)
    rot_mat = np.stack([right, true_up, -forward], axis=1)  # columns = local axes in world
    rot = Rotation.from_matrix(rot_mat)
    return quat_wxyz_from_rotation(rot)


def hump(phase01: float) -> float:
    """Smooth single bump 0 -> 1 -> 0 over phase in [0, 1)."""
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * (phase01 % 1.0))


def triangle(phase01: float) -> float:
    """Triangle wave 0 -> 1 -> 0 over phase in [0, 1)."""
    p = phase01 % 1.0
    return 1.0 - abs(2.0 * p - 1.0)


def smoothstep(x: float) -> float:
    """Cubic ease 0 -> 1 over x in [0, 1], with zero velocity at both ends
    -- used to ease acceleration/deceleration of staged trajectories so
    walks start/stop smoothly instead of snapping to a constant speed.
    """
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


# --------------------------------------------------------------------------
# Loading flyrl modules without importing the real `flyrl` package
# (flyrl/__init__.py pulls in fastbrain.py, which needs torch; torch is
# intentionally NOT installed in .venv-body). We inject a stub `flyrl`
# package into sys.modules and load addiction_env.py / scripted.py from
# their files directly, so their own `from flyrl.addiction_env import ...`
# statements resolve against the stub instead of re-running __init__.py.
# --------------------------------------------------------------------------

def load_flyrl_modules():
    flyrl_dir = os.path.join(_PROJECT_ROOT, "flyrl")
    if "flyrl" not in sys.modules:
        stub = types.ModuleType("flyrl")
        stub.__path__ = [flyrl_dir]
        sys.modules["flyrl"] = stub

    def _load(modname: str):
        full = f"flyrl.{modname}"
        if full in sys.modules:
            return sys.modules[full]
        path = os.path.join(flyrl_dir, f"{modname}.py")
        spec = importlib.util.spec_from_file_location(full, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        setattr(sys.modules["flyrl"], modname, mod)
        return mod

    addiction_env = _load("addiction_env")
    scripted = _load("scripted")
    return addiction_env, scripted


# --------------------------------------------------------------------------
# Joint names
# --------------------------------------------------------------------------

LEG_SEGMENTS = ("T1", "T2", "T3")
SIDES = ("left", "right")

# Alternating tripod grouping: (T1_left, T2_right, T3_left) vs
# (T1_right, T2_left, T3_right) -- standard insect tripod gait.
TRIPOD_GROUP = {
    ("T1", "left"): 0, ("T2", "right"): 0, ("T3", "left"): 0,
    ("T1", "right"): 1, ("T2", "left"): 1, ("T3", "right"): 1,
}

WING_JOINTS = [f"wing_{dof}_{side}" for side in SIDES for dof in ("yaw", "roll", "pitch")]

MOUTH_JOINTS = ["rostrum", "haustellum", "haustellum_abduct", "labrum_left", "labrum_right"]

SWIPE_LEG = ("T1", "right")   # foreleg used to swipe the phone screen

# --------------------------------------------------------------------------
# Gait / animation tuning constants (radians, seconds, cm)
# --------------------------------------------------------------------------

STRIDE_HZ = 2.4
COXA_AMP = 0.34
FEMUR_AMP = 0.22
BOB_AMP = 0.006

DRAG_PERIOD = 1.6          # seconds per smoking "drag" cycle
ROSTRUM_EXTEND_SMOKE = 0.95
EXHALE_TRIGGER_PHASE = 0.5  # puff fires just after the peak (start of exhale)

EAT_PERIOD = 0.7
ROSTRUM_EXTEND_EAT = 1.0

SCROLL_PERIOD = 1.0
SWIPE_COXA_AMP = 0.55
SWIPE_FEMUR_LIFT_AMP = 0.55
SWIPE_TIBIA_AMP = 0.35

N_SMOKE = 40
SMOKE_RISE = 0.09           # cm/s
SMOKE_DRIFT = 0.035         # cm/s
SMOKE_GROW = 0.045          # cm/s radius growth (slow: puffs stay small wisps, never engulf the head)
SMOKE_FADE = 0.80           # alpha/s (fade out while still small)
HEAD_WIDTH = 0.28 * BODY_LENGTH   # rough fly head width, for sizing smoke puffs
SMOKE_R0 = 0.12 * HEAD_WIDTH        # continuous tip wisps: start tiny
SMOKE_ALPHA0 = 0.34                 # continuous tip wisps: low initial alpha
SMOKE_R0_BIG = 0.18 * HEAD_WIDTH     # exhale puff: "~0.15-0.2 head-widths" per spec
SMOKE_ALPHA0_BIG = 0.45             # exhale puff: low initial alpha, not a solid blob
SMOKE_JITTER_STD = 0.35 * HEAD_WIDTH  # spatial spread so a burst reads as several distinct wisps
SMOKE_MAX_R = 0.55 * HEAD_WIDTH       # hard cap: a puff sphere must never get head-sized

REELS_SLIDE_DUR = 0.3     # seconds: ease-out feed-scroll transition duration
REELS_JACKPOT_LIFE = 0.5  # seconds: big centre heart pop duration


# --------------------------------------------------------------------------
# Building the combined MJCF scene: fly + arena + props + dynamic fx geoms
# --------------------------------------------------------------------------

def _build_fly() -> mjcf.RootElement:
    fly = mjcf.from_path(FLY_XML)
    thorax = fly.find("body", "thorax")
    thorax.find("joint", "free").remove()
    return fly


@dataclass
class PropIds:
    """Cached geom/body ids and qpos addresses used every frame."""
    food_body: int
    cigarette_body: int
    ashtray_body: int
    tip_geom: int
    phone_body: int
    screen_geom: int
    smoke_geoms: list
    cam_ids: dict
    tip_mat: int
    screen_mat: int
    reels_tex_id: int = -1
    screen_basis: dict = field(default_factory=dict)
    # geometric constants needed to recompute derived world frames
    # after a prop is moved (see `recompute_prop_frames`).
    cig_len: float = 0.0
    cig_r: float = 0.0
    r_food: float = 0.0
    ph_h: float = 0.0
    phone_prop_angle: float = math.radians(60)
    tip_world: np.ndarray = None
    filter_world: np.ndarray = None
    food_world: np.ndarray = None


def _add_scene_assets(root: mjcf.RootElement):
    root.asset.add("texture", name="skybox", type="skybox", builtin="gradient",
                    rgb1=[0.55, 0.68, 0.85], rgb2=[0.08, 0.08, 0.12],
                    width=200, height=200)
    root.asset.add("texture", name="grid", type="2d", builtin="checker",
                    rgb1=[0.5, 0.53, 0.58], rgb2=[0.58, 0.61, 0.66],
                    width=300, height=300, mark="edge", markrgb=[0.62, 0.65, 0.7])
    root.asset.add("material", name="grid", texture="grid",
                    texrepeat=[8, 8], texuniform=True, reflectance=0.02,
                    specular=0.0, shininess=0.0)
    root.asset.add("material", name="amber", rgba=[1.0, 0.72, 0.15, 0.55],
                    specular=0.6, shininess=0.7, reflectance=0.15)
    root.asset.add("material", name="amber_hi", rgba=[1.0, 0.92, 0.7, 0.75],
                    specular=0.8, shininess=0.9)
    root.asset.add("material", name="paper_white", rgba=[0.96, 0.95, 0.90, 1.0],
                    specular=0.1, shininess=0.05)
    root.asset.add("material", name="filter_tan", rgba=[0.82, 0.65, 0.40, 1.0],
                    specular=0.05, shininess=0.05)
    root.asset.add("material", name="ash_gray", rgba=[0.18, 0.18, 0.19, 1.0],
                    specular=0.2, shininess=0.2)
    mat_tip = root.asset.add("material", name="tip_glow", rgba=[1.0, 0.45, 0.05, 1.0],
                              emission=1.0, specular=0.0, shininess=0.0)
    root.asset.add("material", name="phone_body", rgba=[0.08, 0.08, 0.10, 1.0],
                    specular=0.6, shininess=0.7)
    # Placeholder flat texture at the exact reels frame resolution; its
    # pixels get fully overwritten every frame via model.tex_data +
    # mujoco.mjr_uploadTexture (see AddictionScene3D._update_reels_screen).
    root.asset.add("texture", name="reels_screen_tex", type="2d", builtin="flat",
                    width=reels_media.REELS_W, height=reels_media.REELS_H,
                    rgb1=[0.05, 0.05, 0.06])
    mat_screen = root.asset.add("material", name="screen_glow", rgba=[1.0, 1.0, 1.0, 1.0],
                                 texture="reels_screen_tex", texuniform=False,
                                 emission=0.7, specular=0.05, shininess=0.05)
    return mat_tip, mat_screen


def build_scene():
    """Build the full MJCF model: fly + floor arena + food/cigarette/phone
    props + a pool of dynamic "fx" geoms (smoke particles, scroll cards,
    heart) that get repositioned/recoloured every frame from Python.

    Returns (physics, ids) where `physics` is a `dm_control.mjcf.Physics`
    (wrapping raw mujoco.MjModel / MjData at `.model.ptr` / `.data.ptr`)
    and `ids` is a `PropIds` with cached ids/addresses.
    """
    root = mjcf.RootElement(model="fly_addiction_scene")
    mat_tip, mat_screen = _add_scene_assets(root)

    half = ARENA_SCALE * 0.75 + 2.0 * BODY_LENGTH
    root.worldbody.add("geom", name="floor", type="plane",
                        size=[half, half, 0.05], material="grid", pos=[0, 0, FLOOR_Z])

    # soft, mostly-overhead lighting
    root.worldbody.add("light", name="sun", pos=[0, 0, 3.0], dir=[0.15, 0.1, -1],
                        diffuse=[0.55, 0.55, 0.53], specular=[0.08, 0.08, 0.08],
                        ambient=[0.30, 0.30, 0.33], directional=True, castshadow=True)
    root.worldbody.add("light", name="fill", pos=[-1.2, -1.2, 1.0], dir=[0.6, 0.6, -0.5],
                        diffuse=[0.22, 0.22, 0.24], directional=True, castshadow=False)
    root.worldbody.add("light", name="rim", pos=[1.0, 1.0, 0.6], dir=[-0.6, -0.6, -0.2],
                        diffuse=[0.15, 0.15, 0.17], directional=True, castshadow=False)

    # ---- fly ----
    fly = _build_fly()
    frame = root.attach(fly)
    frame.pos = [0, 0, 0]
    frame.add("freejoint", name="fly_free")

    # ---- FOOD: translucent amber sugar droplet ----
    r_food = 0.42 * BODY_LENGTH
    food_body = root.worldbody.add("body", name="food_body", pos=[0.6 * ARENA_SCALE, 0, FLOOR_Z])
    food_body.add("geom", name="food_drop", type="sphere", size=[r_food, 0, 0],
                   material="amber", pos=[0, 0, r_food * 0.75])
    food_body.add("geom", name="food_hi", type="sphere", size=[r_food * 0.35, 0, 0],
                   material="amber_hi", pos=[r_food * 0.25, -r_food * 0.2, r_food * 1.15])

    # ---- SMOKE: cigarette lying propped at an angle + ashtray ----
    cig_len = 2.6 * BODY_LENGTH
    cig_r = 0.09 * BODY_LENGTH
    filter_frac = 0.34
    tip_r = cig_r * 1.35
    ash_len = cig_r * 3.0   # thin grey ash segment behind the ember

    ash_r = 1.15 * BODY_LENGTH
    ash_h = 0.06 * BODY_LENGTH

    # The env's `smoke` source point is where the fly is drawn to stand
    # (within r_consume). We want that point to coincide with the
    # *filter* end (mouth contact), at fly-head height -- not the ashtray
    # -- so the cigarette body is placed such that its filter tip lands
    # exactly at (smoke_xy, FILTER_TARGET_Z); the ashtray then sits under
    # wherever the (low, glowing) tip end of that geometry falls out. The
    # cigarette's yaw is chosen (see `cigarette_pose`) so its long axis
    # points from the ashtray inward toward the arena centre -- the
    # direction a fly is expected to approach from -- so the fly can
    # stand in line with the cigarette and face straight down it.
    smoke_xy0 = np.array([-0.55 * ARENA_SCALE, 0.55 * ARENA_SCALE])
    cig_yaw0 = math.atan2(-smoke_xy0[1], -smoke_xy0[0])
    cig_pos0, cig_quat0 = cigarette_pose(smoke_xy0, cig_len, cig_r, cig_yaw0)
    ashtray_body = root.worldbody.add(
        "body", name="ashtray_body", pos=[cig_pos0[0], cig_pos0[1], FLOOR_Z])
    ashtray_body.add("geom", name="ashtray_disc", type="cylinder",
                      size=[ash_r, ash_h, 0], material="ash_gray", pos=[0, 0, ash_h])
    ashtray_body.add("geom", name="ashtray_rim", type="cylinder",
                      size=[ash_r * 1.15, ash_h * 0.4, 0], material="ash_gray",
                      pos=[0, 0, ash_h * 0.4])

    # cigarette: propped so the tip rests near the ashtray (low) and the
    # filter end rises up to fly-head height, exactly above the smoke
    # source point.
    cig_body = root.worldbody.add(
        "body", name="cigarette_body", pos=list(cig_pos0), quat=list(cig_quat0))
    # local +x runs along the cigarette from tip end (x=0) to filter end (x=cig_len)
    paper_len = cig_len * (1 - filter_frac)
    filt_len = cig_len * filter_frac
    cig_body.add("geom", name="cig_ash", type="capsule",
                  fromto=[cig_r * 0.9, 0, 0, cig_r * 0.9 + ash_len, 0, 0],
                  size=[cig_r * 0.97, 0, 0], material="ash_gray")
    cig_body.add("geom", name="cig_paper", type="capsule",
                  fromto=[cig_r * 0.9 + ash_len, 0, 0, paper_len - cig_r, 0, 0],
                  size=[cig_r, 0, 0], material="paper_white")
    cig_body.add("geom", name="cig_filter", type="capsule",
                  fromto=[paper_len, 0, 0, cig_len - cig_r, 0, 0],
                  size=[cig_r * 1.05, 0, 0], material="filter_tan")
    tip_geom = cig_body.add("geom", name="cig_tip", type="sphere",
                             size=[tip_r, 0, 0], pos=[0, 0, 0], material="tip_glow")
    cig_body.add("light", name="tip_light", pos=[0, 0, tip_r * 2], dir=[0, 0, -1],
                 diffuse=[0.9, 0.45, 0.1], specular=[0.2, 0.1, 0.0],
                 attenuation=[1, 2, 6], directional=False, castshadow=False)

    # ---- REELS: smartphone propped up at ~60 degrees ----
    ph_h = 2.3 * BODY_LENGTH   # long axis (screen height)
    ph_w = 1.05 * BODY_LENGTH  # width
    ph_t = 0.16 * BODY_LENGTH  # thickness
    prop_angle = math.radians(60)  # elevation of the phone's long axis from horizontal
    phone_xy0 = np.array([0.0, -0.62 * ARENA_SCALE])
    # face the phone back toward the arena centre / cigarette+food side
    face_yaw0 = math.atan2(-phone_xy0[1], -phone_xy0[0])
    phone_pos0, phone_quat0 = phone_pose(phone_xy0, ph_h, prop_angle, face_yaw0)
    phone_body = root.worldbody.add(
        "body", name="phone_body", pos=list(phone_pos0), quat=list(phone_quat0))
    phone_body.add("geom", name="phone_case", type="box",
                    size=[ph_w / 2, ph_h / 2, ph_t / 2], material="phone_body")
    screen_geom = phone_body.add(
        "geom", name="phone_screen", type="box",
        size=[ph_w * 0.44, ph_h * 0.46, ph_t * 0.12],
        pos=[0, 0.02 * ph_h, ph_t / 2 + ph_t * 0.13], material="screen_glow")

    # ---- dynamic fx pool: smoke particles (reels cards/heart are now a
    # real video texture on the phone screen itself -- see reels_ui.py) ----
    fx = root.worldbody.add("body", name="fx", pos=[0, 0, 0])
    smoke_geoms = []
    for i in range(N_SMOKE):
        g = fx.add("geom", name=f"smoke_{i}", type="sphere", size=[0.001, 0, 0],
                   pos=[0, 0, -5], rgba=[0.6, 0.6, 0.62, 0.0], contype=0, conaffinity=0)
        smoke_geoms.append(g)

    # ---- cameras (poses overwritten every frame from Python) ----
    cam_orbit = root.worldbody.add("camera", name="cam_orbit", pos=[0.5, -0.5, 0.4],
                                    xyaxes=[1, 1, 0, -0.3, 0.3, 1])
    cam_closeup = root.worldbody.add("camera", name="cam_closeup", pos=[0.3, -0.3, 0.3],
                                      xyaxes=[1, 1, 0, -0.3, 0.3, 1])
    cam_top = root.worldbody.add("camera", name="cam_top", pos=[0, 0, 3],
                                  xyaxes=[1, 0, 0, 0, 1, 0])

    physics = mjcf.Physics.from_mjcf_model(root)

    def gid(name):
        i = mujoco.mj_name2id(physics.model.ptr, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert i >= 0, f"geom not found: {name}"
        return i

    def bid(name):
        i = mujoco.mj_name2id(physics.model.ptr, mujoco.mjtObj.mjOBJ_BODY, name)
        assert i >= 0, f"body not found: {name}"
        return i

    def matid(name):
        i = mujoco.mj_name2id(physics.model.ptr, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        assert i >= 0, f"material not found: {name}"
        return i

    def camid(name):
        i = mujoco.mj_name2id(physics.model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, name)
        assert i >= 0, f"camera not found: {name}"
        return i

    def texid(name):
        i = mujoco.mj_name2id(physics.model.ptr, mujoco.mjtObj.mjOBJ_TEXTURE, name)
        assert i >= 0, f"texture not found: {name}"
        return i

    ids = PropIds(
        food_body=bid("food_body"),
        cigarette_body=bid("cigarette_body"),
        ashtray_body=bid("ashtray_body"),
        tip_geom=gid("cig_tip"),
        phone_body=bid("phone_body"),
        screen_geom=gid("phone_screen"),
        smoke_geoms=[gid(f"smoke_{i}") for i in range(N_SMOKE)],
        cam_ids={"orbit": camid("cam_orbit"), "closeup": camid("cam_closeup"),
                 "top": camid("cam_top")},
        tip_mat=matid("tip_glow"),
        screen_mat=matid("screen_glow"),
        reels_tex_id=texid("reels_screen_tex"),
        cig_len=cig_len, cig_r=cig_r, r_food=r_food, ph_h=ph_h,
    )
    ids.screen_basis["half_h"] = ph_h * 0.46
    ids.screen_basis["half_w"] = ph_w * 0.44

    m, d = physics.model.ptr, physics.data.ptr
    # Tone down MuJoCo's default camera headlight so lighting reads as the
    # soft scene lights above, not a hot spot centred on whatever the
    # active camera looks at.
    m.vis.headlight.ambient[:] = [0.0, 0.0, 0.0]
    m.vis.headlight.diffuse[:] = [0.0, 0.0, 0.0]
    m.vis.headlight.specular[:] = [0.0, 0.0, 0.0]
    # The fruitfly asset itself ships 3 `trackcom` lights (right/left/
    # tracking) that always follow the fly and produce a bright halo
    # centred on it wherever the camera looks. We don't modify the
    # flybody asset file, but we can zero these out on our own compiled
    # model instance so our own scene lights (above) are what's seen.
    for lname in ("fruitfly/right", "fruitfly/left", "fruitfly/tracking"):
        lid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, lname)
        if lid >= 0:
            m.light_diffuse[lid] = [0.0, 0.0, 0.0]
            m.light_specular[lid] = [0.0, 0.0, 0.0]
            m.light_ambient[lid] = [0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    recompute_prop_frames(m, d, ids)

    return physics, ids


def recompute_prop_frames(m, d, ids: PropIds):
    """(Re)derive world-space anchor points/bases for the props from the
    current `d.xpos`/`d.xmat` of their (possibly just-moved) bodies.
    Must be called after any change to `model.body_pos`/`body_quat` for
    the prop bodies, following an `mj_forward`.
    """
    R = d.xmat[ids.phone_body].reshape(3, 3)
    screen_local_pos = m.geom_pos[ids.screen_geom]
    screen_center = d.xpos[ids.phone_body] + R @ screen_local_pos
    ids.screen_basis.update({
        "center": screen_center,
        "right": R[:, 0].copy(),
        "up": R[:, 1].copy(),
        "normal": R[:, 2].copy(),
    })
    tip_R = d.xmat[ids.cigarette_body].reshape(3, 3)
    ids.tip_world = d.xpos[ids.cigarette_body] + tip_R @ m.geom_pos[ids.tip_geom]
    filt_local = np.array([ids.cig_len - ids.cig_r, 0, 0])
    ids.filter_world = d.xpos[ids.cigarette_body] + tip_R @ filt_local
    ids.food_world = d.xpos[ids.food_body] + np.array([0, 0, ids.r_food * 0.75])


def env_to_world(xy) -> np.ndarray:
    """Map env arena coords in [0,1]^2 to world (X, Y) centred at origin."""
    xy = np.asarray(xy, dtype=np.float64)
    return (xy - 0.5) * ARENA_SCALE


def world_to_env(xy) -> np.ndarray:
    """Inverse of `env_to_world`."""
    xy = np.asarray(xy, dtype=np.float64)
    return xy / ARENA_SCALE + 0.5


FILTER_TARGET_Z = 0.062   # filter (mouth contact) height: measured to match the
                          # fly's own extended-proboscis height (it droops down
                          # as it extends -- see `_mouth_world`), not head height
TIP_TARGET_Z = 0.03                # tip (ember) height: resting just above the ashtray surface


def cigarette_pose(smoke_world_xy, cig_len: float, cig_r: float,
                   yaw: float, tilt: Optional[float] = None):
    """World (position, quat) for the cigarette body such that its
    *filter* tip (not its body origin / burning tip) lands exactly at
    (smoke_world_xy, FILTER_TARGET_Z) -- i.e. at fly-head height, right
    at the env source point the fly is drawn to -- with the cigarette's
    long axis (tip -> filter) pointing along `yaw` (world bearing, radians).

    `tilt` (rise from horizontal, radians) defaults to whatever angle
    makes the *tip* end land at TIP_TARGET_Z (near the ashtray surface),
    so the prop angle self-adjusts correctly if `cig_len` changes instead
    of leaving the tip floating or buried underfloor.
    """
    L = cig_len - cig_r
    if tilt is None:
        tilt = math.asin(float(np.clip((FILTER_TARGET_Z - TIP_TARGET_Z) / L, -1.0, 1.0)))
    # Intrinsic order matters here (unlike a free camera look-at): tilt
    # UP first (about local/world Y, since no yaw applied yet), THEN yaw
    # about world Z -- so the vertical rise stays sin(tilt) regardless of
    # yaw. (`Rotation.from_euler("zy", [yaw, -tilt])` would instead yaw
    # first, coupling the height into a cos(yaw) factor.)
    rot = Rotation.from_euler("z", yaw) * Rotation.from_euler("y", -tilt)
    axis = rot.apply([1.0, 0.0, 0.0])  # world dir from tip (local x=0) to filter (local x=L)
    smoke_world_xy = np.asarray(smoke_world_xy, dtype=np.float64)
    filter_target = np.array([smoke_world_xy[0], smoke_world_xy[1], FILTER_TARGET_Z])
    body_pos = filter_target - axis * L
    body_quat = quat_wxyz_from_rotation(rot)
    return body_pos, body_quat


def phone_pose(reels_world_xy, ph_h: float, prop_angle: float, yaw: float):
    """World (position, quat) for the phone body, built by constructing
    its local (right, height, normal) basis directly from `yaw`/`prop_angle`
    rather than composing Euler angles -- an earlier `Rotation.from_euler
    ("zy", [yaw, pitch])` version had two yaw-coupling bugs that only
    showed up for certain yaw values: the screen normal didn't actually
    rotate with `yaw` at all (always faced the same fixed world
    direction), and the screen's "up" axis silently flipped sign (upside
    down video/UI) for yaw < 0. Both are avoided by building the basis
    from independent, always-orthonormal vectors:

      fwd        = horizontal bearing `yaw` (unit vector in the XY plane)
      height_dir = -cos(prop_angle)*fwd + sin(prop_angle)*world_up
                   (the phone's long axis; leans back away from `fwd` as
                   prop_angle drops from 90 deg (bolt upright))
      normal_dir =  sin(prop_angle)*fwd + cos(prop_angle)*world_up
                   (the screen's outward-facing normal; always has a
                   positive `fwd` component and a positive-yaw-independent
                   upward tilt)

    The phone stands with its base at (reels_world_xy, floor), rising by
    ph_h/2 along `height_dir` to its centre (the body origin).
    """
    reels_world_xy = np.asarray(reels_world_xy, dtype=np.float64)
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    world_up = np.array([0.0, 0.0, 1.0])
    height_dir = -math.cos(prop_angle) * fwd + math.sin(prop_angle) * world_up
    normal_dir = math.sin(prop_angle) * fwd + math.cos(prop_angle) * world_up
    right_dir = np.cross(height_dir, normal_dir)
    right_dir = right_dir / (np.linalg.norm(right_dir) + 1e-12)

    rot_mat = np.stack([right_dir, height_dir, normal_dir], axis=1)  # columns = local X,Y,Z in world
    body_quat = quat_wxyz_from_rotation(Rotation.from_matrix(rot_mat))
    body_pos = np.array([reels_world_xy[0], reels_world_xy[1], FLOOR_Z]) + height_dir * (ph_h / 2.0)
    return body_pos, body_quat


# --------------------------------------------------------------------------
# Main scene / animation controller
# --------------------------------------------------------------------------

ACTIVITY_LABELS = {
    "food": "eating",
    "smoke": "smoking",
    "reels": "scrolling reels",
    "snub": "ignoring food",
}


class AddictionScene3D:
    """Kinematic MuJoCo scene of the addiction-toy fly: call `set_sources`
    once (or whenever the env resets), then `set_state` every env step (or
    every interpolated sub-frame) followed by `render(camera=...)`.
    """

    def __init__(self, width: int = RENDER_W, height: int = RENDER_H,
                create_renderer: bool = True):
        self.physics, self.ids = build_scene()
        self.m = self.physics.model.ptr
        self.d = self.physics.data.ptr
        self.width, self.height = width, height
        # `create_renderer=False` skips the offscreen mujoco.Renderer /
        # GL framebuffer -- used by view3d.py's interactive passive viewer,
        # which drives its own on-screen GL context and doesn't need ours.
        self.renderer = mujoco.Renderer(self.m, height=height, width=width) \
            if create_renderer else None

        self._qadr = {}
        for seg in LEG_SEGMENTS:
            for side in SIDES:
                for base in ("coxa", "femur", "tibia", "tarsus"):
                    name = f"{base}_{seg}_{side}"
                    self._qadr[name] = self._joint_qadr(name)
        for name in MOUTH_JOINTS + WING_JOINTS + ["head", "head_twist", "head_abduct"]:
            self._qadr[name] = self._joint_qadr(name)
        free_jids = [j for j in range(self.m.njnt)
                    if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        assert len(free_jids) == 1, f"expected exactly one freejoint, found {len(free_jids)}"
        self._free_qadr = int(self.m.jnt_qposadr[free_jids[0]])

        self._init_standing_pose()

        # animation / bookkeeping state
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._t = 0.0
        self._prev_t = 0.0
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._speed_norm = 0.0
        self._at = None
        self._prev_at = None
        self._activity_t0 = 0.0
        self._prev_smoke_phase = 0.0
        self._prev_scroll_phase = 0.0

        # ---- reels video state ----
        self._reels_names, self._reels_frames = reels_media.load_reels_clips(REELS_ASSETS_DIR)
        self._reels_current_i = 0
        self._reels_cycle_t0 = 0.0
        self._reels_prev_phase = 0.0
        self._reels_video_clock = 0.0
        self._reels_jackpot_t0 = -10.0
        self._reels_last_rgb = None
        self._reels_tex_dirty = False

        self._smoke = {
            "pos": np.zeros((N_SMOKE, 3)),
            "vel": np.zeros((N_SMOKE, 3)),
            "r": np.zeros(N_SMOKE),
            "alpha": np.zeros(N_SMOKE),
            "alive": np.zeros(N_SMOKE, dtype=bool),
        }
        self._rng = np.random.default_rng(0)

        try:
            self._font = ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", 18)
            self._font_small = ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            self._font = ImageFont.load_default()
            self._font_small = self._font

        mujoco.mj_forward(self.m, self.d)

    # ------------------------------------------------------------------
    def _joint_qadr(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"fruitfly/{name}")
        assert jid >= 0, f"joint not found: {name}"
        return self.m.jnt_qposadr[jid]

    def _init_standing_pose(self):
        d = self.d
        # legs: the model's rest pose (qpos == 0) is already a plausible
        # standing stance (verified visually) -- nothing to do there.
        # wings: fold them using the spring-reference pose, the same trick
        # flybody itself uses to retract unused wings.
        for name in WING_JOINTS:
            qadr = self._qadr[name]
            d.qpos[qadr] = self.m.qpos_spring[qadr]
        for name in MOUTH_JOINTS + ["head", "head_twist", "head_abduct"]:
            d.qpos[self._qadr[name]] = 0.0

    # ------------------------------------------------------------------
    def set_sources(self, food_xy, smoke_xy, reels_xy):
        """Reposition FOOD / SMOKE(cigarette+ashtray) / REELS(phone) props
        from env-space [0,1]^2 source coordinates.
        """
        m, d = self.m, self.d
        food_w = env_to_world(food_xy)
        smoke_w = env_to_world(smoke_xy)
        reels_w = env_to_world(reels_xy)

        m.body_pos[self.ids.food_body][:2] = food_w
        cig_yaw = math.atan2(-smoke_w[1], -smoke_w[0])
        cig_pos, cig_quat = cigarette_pose(smoke_w, self.ids.cig_len, self.ids.cig_r, cig_yaw)
        m.body_pos[self.ids.cigarette_body][:] = cig_pos
        m.body_quat[self.ids.cigarette_body][:] = cig_quat
        m.body_pos[self.ids.ashtray_body][:2] = cig_pos[:2]

        face_yaw = math.atan2(-reels_w[1], -reels_w[0])
        phone_pos, phone_quat = phone_pose(reels_w, self.ids.ph_h,
                                           self.ids.phone_prop_angle, face_yaw)
        m.body_pos[self.ids.phone_body][:] = phone_pos
        m.body_quat[self.ids.phone_body][:] = phone_quat

        mujoco.mj_forward(m, d)
        recompute_prop_frames(m, d, self.ids)

    # ------------------------------------------------------------------
    def set_state(self, x: float, y: float, heading: float, at: Optional[str],
                  hunger: float, nicotine: float, tolerance: float,
                  jackpot: bool, t: float):
        """Set the fly's pose and prop animation state for time `t` (secs).

        x, y: env-space position in [0, 1]^2.
        heading: env-space heading, radians.
        at: one of {'food', 'smoke', 'reels', None}.
        hunger, nicotine, tolerance: in [0, 1], drawn on the HUD.
        jackpot: True on the (single) step a reels jackpot just fired.
        t: animation clock, seconds (monotonically increasing).
        """
        m, d = self.m, self.d
        dt = max(1e-4, min(0.5, t - self._prev_t))
        world_xy = env_to_world((x, y))

        dx, dy = world_xy[0] - self._prev_x, world_xy[1] - self._prev_y
        dist = math.hypot(dx, dy)
        raw_speed = dist / dt
        target_speed_norm = float(np.clip(raw_speed / (0.35 * ARENA_SCALE), 0.0, 1.0))
        # smooth speed estimate so gait amplitude doesn't chatter frame-to-frame
        self._speed_norm = 0.8 * self._speed_norm + 0.2 * target_speed_norm
        if at is not None:
            self._speed_norm = 0.0  # parked at a source: legs settle to stance

        if at != self._prev_at:
            self._activity_t0 = t
        self._prev_at = self._at
        self._at = at
        self._x, self._y, self._heading = x, y, heading
        self._t = t
        self._last_hunger = hunger
        self._last_nicotine = nicotine
        self._last_tolerance = tolerance

        # ---- root freejoint: position + yaw ----
        bob = BOB_AMP * self._speed_norm * abs(math.sin(4.0 * math.pi * STRIDE_HZ * t))
        d.qpos[self._free_qadr + 0] = world_xy[0]
        d.qpos[self._free_qadr + 1] = world_xy[1]
        d.qpos[self._free_qadr + 2] = SPAWN_Z + bob
        d.qpos[self._free_qadr + 3: self._free_qadr + 7] = quat_from_yaw(heading)

        self._update_gait(t)
        self._update_mouth(t, at)
        self._update_swipe(t, at)
        self._update_smoke_particles(dt, t, at)
        self._update_reels_screen(dt, t, at, jackpot)
        self._update_tip_glow(t)

        self._prev_t = t
        self._prev_x, self._prev_y = world_xy

        mujoco.mj_forward(m, d)

    # ------------------------------------------------------------------
    def _update_gait(self, t: float):
        d = self.d
        swipe_seg, swipe_side = SWIPE_LEG
        for seg in LEG_SEGMENTS:
            for side in SIDES:
                coxa_name = f"coxa_{seg}_{side}"
                femur_name = f"femur_{seg}_{side}"
                if (seg, side) == (swipe_seg, swipe_side) and self._at == "reels":
                    continue  # driven by _update_swipe instead
                group = TRIPOD_GROUP[(seg, side)]
                phase = group * math.pi + 2.0 * math.pi * STRIDE_HZ * t
                d.qpos[self._qadr[coxa_name]] = COXA_AMP * self._speed_norm * math.sin(phase)
                d.qpos[self._qadr[femur_name]] = FEMUR_AMP * self._speed_norm * math.cos(phase)
                d.qpos[self._qadr[f"tibia_{seg}_{side}"]] = 0.0

    def _update_mouth(self, t: float, at: Optional[str]):
        d = self.d
        rostrum = 0.0
        if at == "smoke":
            phase = ((t - self._activity_t0) / DRAG_PERIOD) % 1.0
            rostrum = -ROSTRUM_EXTEND_SMOKE * hump(phase)
            if self._prev_smoke_phase < EXHALE_TRIGGER_PHASE <= phase + 1e-6:
                self._spawn_smoke_burst(self._mouth_world(), n=8, big=True)
            self._prev_smoke_phase = phase
        elif at == "food":
            phase = ((t - self._activity_t0) / EAT_PERIOD) % 1.0
            rostrum = -ROSTRUM_EXTEND_EAT * hump(phase)
        else:
            self._prev_smoke_phase = 0.0
        d.qpos[self._qadr["rostrum"]] = rostrum
        d.qpos[self._qadr["haustellum"]] = -0.5 * max(0.0, -rostrum)
        d.qpos[self._qadr["labrum_left"]] = 0.5 * max(0.0, -rostrum)
        d.qpos[self._qadr["labrum_right"]] = 0.5 * max(0.0, -rostrum)
        # tilt head down slightly toward whatever it's using -- "snub" gets
        # the same small dip as an actual "considering it" cue, even
        # though (unlike "food") it never extends the rostrum.
        d.qpos[self._qadr["head"]] = -0.15 if at in ("food", "smoke", "snub") else 0.0

    def _update_swipe(self, t: float, at: Optional[str]):
        d = self.d
        seg, side = SWIPE_LEG
        if at != "reels":
            # Not scrolling: leave this leg exactly as `_update_gait` just
            # set it (normal walking gait, or the resting stance if
            # speed_norm == 0) -- don't clobber it back to a fixed zero,
            # or the fly would appear to drag/limp on this leg while walking.
            self._prev_scroll_phase = 0.0
            return
        phase = ((t - self._activity_t0) / SCROLL_PERIOD) % 1.0
        tri = triangle(phase)
        tri_lift = triangle(phase + 0.5)
        d.qpos[self._qadr[f"coxa_{seg}_{side}"]] = SWIPE_COXA_AMP * tri
        d.qpos[self._qadr[f"femur_{seg}_{side}"]] = -SWIPE_FEMUR_LIFT_AMP * tri_lift
        d.qpos[self._qadr[f"tibia_{seg}_{side}"]] = SWIPE_TIBIA_AMP * tri
        self._prev_scroll_phase = phase

    # ------------------------------------------------------------------
    def _head_world(self) -> np.ndarray:
        hid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "fruitfly/head")
        return self.d.xpos[hid].copy()

    def _mouth_world(self) -> np.ndarray:
        """World position of the mouthparts (the `haustellum` body, distal
        to `rostrum`/proximal to the `labrum_*` tips) -- this moves with
        the actual rostrum/haustellum extension, so smoke exhaled "from
        the mouth" tracks the animated proboscis instead of the static
        head centre.
        """
        hid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "fruitfly/haustellum")
        return self.d.xpos[hid].copy()

    def _spawn_smoke_burst(self, origin: np.ndarray, n: int, big: bool = False):
        s = self._smoke
        free = np.where(~s["alive"])[0]
        if len(free) == 0:
            free = np.arange(N_SMOKE)  # steal the oldest slots if saturated
        n = min(n, len(free))
        idx = free[:n]
        r0 = SMOKE_R0_BIG if big else SMOKE_R0
        alpha0 = SMOKE_ALPHA0_BIG if big else SMOKE_ALPHA0
        jitter_std = SMOKE_JITTER_STD * (1.3 if big else 0.6)
        for i in idx:
            jitter = self._rng.normal(0, jitter_std, size=3)
            jitter[2] = abs(jitter[2]) * 0.4
            speed = SMOKE_DRIFT * (1.4 if big else 1.0)
            vel = np.array([self._rng.normal(0, speed), self._rng.normal(0, speed),
                            SMOKE_RISE * (1.2 if big else 1.0) * (0.7 + 0.6 * self._rng.random())])
            pos = origin + jitter
            r = r0 * (0.7 + 0.6 * self._rng.random())
            alpha = alpha0 * (0.8 + 0.4 * self._rng.random())
            if big:
                # Pre-age each wisp by a small random amount so a burst
                # immediately reads as several separated wisps at
                # different stages, not one instantaneous overlapping
                # cluster centred on the origin.
                pre_dt = float(self._rng.uniform(0.0, 0.18))
                pos = pos + vel * pre_dt
                r = r + SMOKE_GROW * pre_dt
                alpha = max(0.08, alpha - SMOKE_FADE * pre_dt)
            s["pos"][i] = pos
            s["vel"][i] = vel
            s["r"][i] = r
            s["alpha"][i] = alpha
            s["alive"][i] = True

    def _update_smoke_particles(self, dt: float, t: float, at: Optional[str]):
        m = self.m
        s = self._smoke
        alive = s["alive"]
        if np.any(alive):
            s["pos"][alive] += s["vel"][alive] * dt
            s["vel"][alive, 0] += self._rng.normal(0, 0.01, size=alive.sum())
            s["vel"][alive, 1] += self._rng.normal(0, 0.01, size=alive.sum())
            s["r"][alive] += SMOKE_GROW * dt
            s["r"] = np.minimum(s["r"], SMOKE_MAX_R)  # never let a puff sphere get head-sized
            s["alpha"][alive] -= SMOKE_FADE * dt
            died = alive & (s["alpha"] <= 0.02)
            s["alive"][died] = False

        if at == "smoke":
            # gentle continuous wisp from the cigarette tip
            if self._rng.random() < min(1.0, dt * 5.0):
                self._spawn_smoke_burst(self.ids.tip_world, n=1, big=False)

        for i, gid in enumerate(self.ids.smoke_geoms):
            if s["alive"][i]:
                m.geom_pos[gid] = s["pos"][i]
                m.geom_size[gid][0] = max(1e-4, s["r"][i])
                m.geom_rgba[gid] = [0.62, 0.62, 0.65, max(0.0, min(0.65, s["alpha"][i]))]
            else:
                m.geom_rgba[gid][3] = 0.0
                m.geom_pos[gid][2] = -5.0

    def _update_reels_screen(self, dt: float, t: float, at: Optional[str], jackpot: bool):
        """Composes the phone screen's video-feed image (current clip,
        scroll-slide transition, UI overlay) into `model.tex_data`. The
        actual GPU re-upload (`mujoco.mjr_uploadTexture` / the passive
        viewer's `update_texture`) happens in the caller (`render()` /
        `view3d.py`), since that call differs between an offscreen
        `mujoco.Renderer` and the interactive passive viewer.
        """
        m = self.m
        if not self._reels_names:
            return  # no clips found under renders/assets/reels -- leave placeholder texture
        n_clips = len(self._reels_names)
        at_phone = at == "reels"

        if at_phone:
            self._reels_video_clock += dt
            phase = ((t - self._activity_t0) / SCROLL_PERIOD) % 1.0
            if self._reels_prev_phase > phase:  # wrapped: a new scroll cycle started
                self._reels_current_i = (self._reels_current_i + 1) % n_clips
                self._reels_cycle_t0 = t
            self._reels_prev_phase = phase

        slide_frac = 0.0
        if at_phone:
            slide_t = t - self._reels_cycle_t0
            if 0.0 <= slide_t < REELS_SLIDE_DUR:
                x = slide_t / REELS_SLIDE_DUR
                slide_frac = 1.0 - (1.0 - x) ** 3  # ease-out cubic

        cur_name = self._reels_names[self._reels_current_i]
        nxt_name = self._reels_names[(self._reels_current_i + 1) % n_clips]
        cur_frames = self._reels_frames[cur_name]
        nxt_frames = self._reels_frames[nxt_name]
        ci = int(self._reels_video_clock * reels_media.REELS_FPS) % len(cur_frames)
        cur_frame = cur_frames[ci]
        if slide_frac > 0.0:
            ni = int(self._reels_video_clock * reels_media.REELS_FPS) % len(nxt_frames)
            nxt_frame = nxt_frames[ni]
        else:
            nxt_frame = cur_frame
        clip_progress = ci / max(1, len(cur_frames) - 1)

        if jackpot and at_phone:
            self._reels_jackpot_t0 = t
        jackpot_age = t - self._reels_jackpot_t0
        heart_active = 0.0 <= jackpot_age < REELS_JACKPOT_LIFE
        jackpot_glow = max(0.0, 1.0 - jackpot_age / REELS_JACKPOT_LIFE) if heart_active else 0.0

        if at_phone or self._reels_last_rgb is None:
            rgb = reels_ui.compose_frame(cur_frame, nxt_frame, slide_frac,
                                         clip_progress, heart_active, jackpot_glow)
            self._reels_last_rgb = rgb
        else:
            rgb = self._reels_last_rgb  # paused: keep showing the last composed frame

        tex_id = self.ids.reels_tex_id
        adr = m.tex_adr[tex_id]
        nc = m.tex_nchannel[tex_id]
        size = reels_media.REELS_H * reels_media.REELS_W * nc
        m.tex_data[adr:adr + size] = rgb.reshape(-1)
        self._reels_tex_dirty = True

        # dim the screen 50% (via the material tint, not the cached pixels)
        # while the fly isn't at the phone.
        m.mat_rgba[self.ids.screen_mat][:3] = 1.0 if at_phone else 0.5

    def _update_tip_glow(self, t: float):
        """Pulses the cigarette ember brighter during each smoking drag."""
        pulse = 0.0
        if self._at == "smoke":
            phase = ((t - self._activity_t0) / DRAG_PERIOD) % 1.0
            pulse = hump(phase)
        self.m.mat_rgba[self.ids.tip_mat] = [1.0, 0.18 + 0.15 * pulse, 0.01, 1.0]
        self.m.mat_emission[self.ids.tip_mat] = 0.8 + 0.8 * pulse

    # ------------------------------------------------------------------
    def _camera_pose(self, name: str):
        fly_pos = np.array([self._x, self._y])
        world_xy = env_to_world(fly_pos)
        heading = self._heading
        forward = np.array([math.cos(heading), math.sin(heading), 0.0])
        body_pos = np.array([world_xy[0], world_xy[1], SPAWN_Z + 0.04])

        if name == "top":
            extent = ARENA_SCALE * 0.75 + 2.0 * BODY_LENGTH
            height = extent * 1.55
            cam_pos = np.array([0.0, 0.0, height])
            target = np.array([0.0, 0.0, FLOOR_Z])
            return cam_pos, target, (1.0, 0.0, 0.0)

        if name == "orbit":
            radius = 0.68
            elev = math.radians(30)
            angle = math.radians(14) * self._t  # slow orbit
            cam_pos = body_pos + np.array([
                radius * math.cos(elev) * math.cos(angle),
                radius * math.cos(elev) * math.sin(angle),
                radius * math.sin(elev),
            ])
            target = body_pos + np.array([0, 0, 0.03])
            return cam_pos, target, (0.0, 0.0, 1.0)

        if name == "closeup":
            if self._at == "smoke":
                # Special, wider framing for smoking: pull back and to the
                # side so the WHOLE cigarette (filter, paper, glowing tip)
                # plus the fly's head and its rising smoke are all in
                # frame -- not just the head/filter contact point.
                tip = self.ids.tip_world
                filt = self.ids.filter_world
                head = self._head_world()
                axis = filt - tip
                axis[2] = 0.0
                axis_n = axis / (np.linalg.norm(axis) + 1e-9)
                side_dir = np.array([-axis_n[1], axis_n[0], 0.0])
                center = 0.5 * (head + tip)
                center[2] = max(head[2], tip[2], filt[2]) * 0.55 + 0.02
                dist = 0.95
                elev = math.radians(24)
                cam_pos = (center + side_dir * dist * math.cos(elev)
                          - axis_n * dist * 0.2
                          + np.array([0.0, 0.0, dist * math.sin(elev)]))
                return cam_pos, center, (0.0, 0.0, 1.0)

            if self._at == "reels":
                # Over-the-shoulder framing: camera sits along the screen's
                # own outward-facing normal (so the screen is seen
                # face-on, undistorted by its 60-degree prop angle),
                # beyond where the fly stands, offset to one side so the
                # fly's head/shoulder sits in the foreground.
                b = self.ids.screen_basis
                center, right = b["center"], b["right"]
                normal_h = b["normal"].copy()
                normal_h[2] = 0.0
                nn = np.linalg.norm(normal_h)
                normal_h = normal_h / nn if nn > 1e-6 else np.array([1.0, 0.0, 0.0])
                dist = 1.25
                elev = math.radians(20)
                cam_pos = (center + normal_h * dist * math.cos(elev) + right * dist * 0.22
                          + np.array([0.0, 0.0, dist * math.sin(elev)]))
                return cam_pos, center, (0.0, 0.0, 1.0)

            prop_point = None
            if self._at in ("food", "snub"):
                prop_point = self.ids.food_world
            head = self._head_world()
            if prop_point is not None:
                target = 0.5 * (head + prop_point)
                mid_dir = prop_point - head
                mid_dir[2] = 0
                n = np.linalg.norm(mid_dir)
                side_dir = np.array([-forward[1], forward[0], 0.0])
                cam_pos = target + side_dir * 0.55 - forward * 0.15 + np.array([0, 0, 0.28])
            else:
                target = head + forward * 0.1
                side_dir = np.array([-forward[1], forward[0], 0.0])
                cam_pos = head - forward * 0.35 + side_dir * 0.35 + np.array([0, 0, 0.28])
            return cam_pos, target, (0.0, 0.0, 1.0)

        raise ValueError(f"unknown camera {name!r}")

    def _auto_camera_name(self) -> str:
        return "closeup" if self._at is not None else "orbit"

    # ------------------------------------------------------------------
    def render(self, camera: str = "orbit") -> np.ndarray:
        if camera == "auto":
            camera = self._auto_camera_name()
        cam_pos, target, up = self._camera_pose(camera)
        quat = look_at_quat(cam_pos, target, world_up=up)
        cam_id = self.ids.cam_ids[camera]
        self.m.cam_pos[cam_id] = cam_pos
        self.m.cam_quat[cam_id] = quat
        mujoco.mj_forward(self.m, self.d)

        if self._reels_tex_dirty and self.renderer is not None:
            mujoco.mjr_uploadTexture(self.m, self.renderer._mjr_context, self.ids.reels_tex_id)
            self._reels_tex_dirty = False

        self.renderer.update_scene(self.d, camera=f"cam_{camera}")
        img = self.renderer.render()
        img = self._draw_hud(img)
        return img

    # ------------------------------------------------------------------
    def _draw_hud(self, img: np.ndarray) -> np.ndarray:
        im = Image.fromarray(img).convert("RGBA")
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        pad = 14
        panel_w, panel_h = 250, 96
        draw.rounded_rectangle([pad, pad, pad + panel_w, pad + panel_h],
                               radius=10, fill=(15, 15, 20, 150))

        bars = [("hunger", self._last_hunger, (230, 140, 40)),
                ("nicotine", self._last_nicotine, (200, 50, 50)),
                ("tolerance", self._last_tolerance, (60, 130, 220))]
        bx, by, bw, bh, gap = pad + 12, pad + 14, panel_w - 24, 14, 8
        for i, (label, val, color) in enumerate(bars):
            y0 = by + i * (bh + gap)
            draw.rectangle([bx, y0, bx + bw, y0 + bh], outline=(230, 230, 230, 220), width=1)
            filled = int(bw * float(np.clip(val, 0.0, 1.0)))
            if filled > 0:
                draw.rectangle([bx, y0, bx + filled, y0 + bh], fill=(*color, 230))
            draw.text((bx, y0 - 13), label, font=self._font_small, fill=(235, 235, 235, 255))

        label = ACTIVITY_LABELS.get(self._at)
        if label is None:
            label = "walking" if self._speed_norm > 0.05 else "idle"
        draw.text((pad + 12, pad + panel_h - 4), label.upper(),
                  font=self._font, fill=(255, 255, 255, 255))

        composed = Image.alpha_composite(im, overlay).convert("RGB")
        return np.asarray(composed)

    # convenience setters used by _draw_hud; set alongside set_state
    _last_hunger = 0.0
    _last_nicotine = 0.0
    _last_tolerance = 0.0

    # ------------------------------------------------------------------
    def _draw_caption(self, img: np.ndarray, title: Optional[str],
                      subtitle: Optional[str] = None) -> np.ndarray:
        """Draws a small two-line caption centred at the bottom of the
        frame: `title` (e.g. "ADDICTED FLY -- trained on hijacked dopamine
        signal") plus a `subtitle` line underneath it (defaults to the
        fixed connectome-provenance line used by the brain-driven renders;
        pass an explicit `subtitle` for other provenance, e.g. the staged
        hand-choreographed demos). Positioned bottom-centre -- clear of the
        HUD panel, which is top-left -- so the two never overlap. No-op if
        `title` is falsy.
        """
        if not title:
            return img
        if subtitle is None:
            subtitle = "steered by a frozen FlyWire connectome (138,639 neurons)"
        im = Image.fromarray(img).convert("RGBA")
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        W, H = im.size

        title_bbox = draw.textbbox((0, 0), title, font=self._font)
        sub_bbox = draw.textbbox((0, 0), subtitle, font=self._font_small)
        title_h = title_bbox[3] - title_bbox[1]
        sub_h = sub_bbox[3] - sub_bbox[1]
        pad_x, pad_y, line_gap = 16, 8, 4
        block_w = max(title_bbox[2] - title_bbox[0], sub_bbox[2] - sub_bbox[0]) + 2 * pad_x
        block_w = min(block_w, W - 24)
        block_h = title_h + sub_h + line_gap + 2 * pad_y

        x0 = max(12.0, (W - block_w) / 2.0)
        y1 = H - 12
        y0 = y1 - block_h
        draw.rounded_rectangle([x0, y0, x0 + block_w, y1], radius=8, fill=(8, 8, 12, 175))
        draw.text((x0 + pad_x, y0 + pad_y - title_bbox[1]), title,
                  font=self._font, fill=(255, 255, 255, 255))
        draw.text((x0 + pad_x, y0 + pad_y + title_h + line_gap - sub_bbox[1]), subtitle,
                  font=self._font_small, fill=(205, 205, 212, 235))

        composed = Image.alpha_composite(im, overlay).convert("RGB")
        return np.asarray(composed)


# --------------------------------------------------------------------------
# Episode rendering (drives the real 2D FlyAddictionEnv + a scripted policy)
# --------------------------------------------------------------------------

def _wrap_angle(theta: float) -> float:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def _make_policy(policy_name: str, seed: int):
    _, scripted = load_flyrl_modules()
    name = policy_name.lower()
    if name == "forager":
        return scripted.ForagerPolicy()
    if name == "smoker":
        return scripted.SmokerPolicy()
    if name == "reels":
        return scripted.ReelsPolicy()
    if name == "greedy":
        return scripted.GreedyPolicy(seed=seed)
    raise ValueError(f"unknown policy_name {policy_name!r}")


def _render_animated(scene: "AddictionScene3D", out: str, step_iter, fps: int = 30,
                     camera: str = "auto", frames_per_step: int = 3,
                     title: Optional[str] = None, max_seconds: Optional[float] = None,
                     init_xyh: Optional[tuple] = None):
    """Shared frame-interpolation + gait/activity-animation + HUD(+caption)
    + mp4-writing loop, used by both `render_episode` (a live scripted-
    policy rollout) and `render_trajectory` (replaying a saved
    flyrl.evaluate trajectory from a trained BrainPolicy).

    `step_iter` yields one dict per env step, in order, each with keys
    x, y, heading (env-space, [0,1]^2 / radians), at, hunger, nicotine,
    tolerance, jackpot -- i.e. exactly `env.step()`'s per-step outputs
    (live or logged). `frames_per_step` sub-frames are interpolated
    between consecutive steps' (x, y, heading), same as the original
    render_episode.

    `init_xyh`, if given, seeds the position interpolated *into* the first
    yielded step (e.g. the env's reset pose, so the very first step's walk
    is visible); otherwise the first step's own (x, y, heading) is used as
    its own predecessor (a static first step, no teleport-in) -- the right
    default when no pre-step-0 pose is available, as with a saved
    trajectory file.
    """
    import imageio

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer = imageio.get_writer(out, fps=fps, codec="libx264",
                                quality=None, bitrate=None,
                                macro_block_size=1,
                                output_params=["-pix_fmt", "yuv420p", "-crf", "20"])

    dt_frame = 1.0 / fps
    t = 0.0
    n_frames = 0
    n_steps = 0
    have_prev = init_xyh is not None
    prev_x, prev_y, prev_h = init_xyh if have_prev else (0.0, 0.0, 0.0)
    try:
        for st in step_iter:
            x, y, heading = float(st["x"]), float(st["y"]), float(st["heading"])
            if not have_prev:
                prev_x, prev_y, prev_h = x, y, heading
                have_prev = True
            dh = _wrap_angle(heading - prev_h)
            n_steps += 1
            stop = False
            for k in range(frames_per_step):
                frac = (k + 1) / frames_per_step
                ix = prev_x + (x - prev_x) * frac
                iy = prev_y + (y - prev_y) * frac
                ih = _wrap_angle(prev_h + dh * frac)
                t += dt_frame
                is_last = (k == frames_per_step - 1)
                scene.set_state(ix, iy, ih, at=st.get("at"),
                                hunger=st.get("hunger", 0.0), nicotine=st.get("nicotine", 0.0),
                                tolerance=st.get("tolerance", 0.0),
                                jackpot=bool(st.get("jackpot", False)) and is_last, t=t)
                frame = scene.render(camera=camera)
                if title:
                    frame = scene._draw_caption(frame, title)
                writer.append_data(frame)
                n_frames += 1
                if max_seconds is not None and t >= max_seconds:
                    stop = True
                    break
            prev_x, prev_y, prev_h = x, y, heading
            if stop:
                break
    finally:
        writer.close()
    return {"out": out, "n_steps": n_steps, "n_frames": n_frames, "duration_s": n_frames / fps}


def render_episode(policy_name: str = "greedy", seed: int = 0,
                   out: str = "renders/episode.mp4", fps: int = 30,
                   camera: str = "auto", frames_per_step: int = 3,
                   scene: Optional["AddictionScene3D"] = None,
                   title: Optional[str] = None, max_seconds: Optional[float] = None):
    """Run one scripted-policy episode of FlyAddictionEnv and render it to
    an mp4, interpolating `frames_per_step` rendered frames per env step
    for smooth motion. `camera='auto'`: orbit while walking, cut to
    closeup while at a source.
    """
    addiction_env, _ = load_flyrl_modules()
    env = addiction_env.FlyAddictionEnv()
    policy = _make_policy(policy_name, seed)

    obs, info = env.reset(seed=seed)
    policy.reset()
    if scene is None:
        scene = AddictionScene3D()
    scene.set_sources(env.sources["food"], env.sources["smoke"], env.sources["reels"])
    init_xyh = (float(env.pos[0]), float(env.pos[1]), float(env.theta))

    state = {"obs": obs, "info": info}

    def step_iter():
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy.act(state["obs"], state["info"])
            obs, reward, terminated, truncated, info = env.step(action)
            if hasattr(policy, "update"):
                policy.update(info.get("at"), reward)
            state["obs"], state["info"] = obs, info
            yield dict(x=float(env.pos[0]), y=float(env.pos[1]), heading=float(env.theta),
                       at=info.get("at"), hunger=info.get("h", 0.0), nicotine=info.get("n", 0.0),
                       tolerance=info.get("tau", 0.0), jackpot=bool(info.get("jackpot", False)))

    return _render_animated(scene, out, step_iter(), fps=fps, camera=camera,
                            frames_per_step=frames_per_step, title=title,
                            max_seconds=max_seconds, init_xyh=init_xyh)


_TRAJ_AT_NAME = {0: None, 1: "food", 2: "smoke", 3: "reels"}


def render_trajectory(traj_path: str, out: str, fps: int = 30, camera: str = "auto",
                      frames_per_step: int = 3, title: Optional[str] = None,
                      max_seconds: Optional[float] = None,
                      scene: Optional["AddictionScene3D"] = None):
    """Replay a saved trajectory (a ``flyrl.evaluate`` ``traj_seed<k>.npz``
    -- see its module docstring for the exact array format: x, y, heading,
    at, h, n, tau, w, jackpot, reward, source_food/smoke/reels) with the
    same per-frame interpolation, walking gait, activity animation and HUD
    as `render_episode` -- but driven by a real trained BrainPolicy's
    logged trajectory (a BRAIN-DRIVEN fly) instead of a live scripted
    policy. `title`, if given (e.g. "ADDICTED FLY -- trained on hijacked
    dopamine signal"), is drawn as a small caption at the bottom of the
    frame alongside a fixed connectome-provenance line -- see
    `AddictionScene3D._draw_caption`.
    """
    data = np.load(traj_path)
    x, y, heading, at_code = data["x"], data["y"], data["heading"], data["at"]
    h, n, tau, jackpot = data["h"], data["n"], data["tau"], data["jackpot"]
    n_steps = len(x)

    if scene is None:
        scene = AddictionScene3D()
    scene.set_sources(data["source_food"], data["source_smoke"], data["source_reels"])

    def step_iter():
        for i in range(n_steps):
            yield dict(x=float(x[i]), y=float(y[i]), heading=float(heading[i]),
                       at=_TRAJ_AT_NAME.get(int(at_code[i])), hunger=float(h[i]),
                       nicotine=float(n[i]), tolerance=float(tau[i]),
                       jackpot=bool(jackpot[i]))

    return _render_animated(scene, out, step_iter(), fps=fps, camera=camera,
                            frames_per_step=frames_per_step, title=title,
                            max_seconds=max_seconds)


# --------------------------------------------------------------------------
# Staged demo / stills (hand-authored trajectory, no 2D env needed)
# --------------------------------------------------------------------------

_STAGE_FOOD_XY = (0.78, 0.5)
_STAGE_SMOKE_XY = (0.22, 0.75)
_STAGE_REELS_XY = (0.5, 0.18)
_STAGE_START_XY = (0.5, 0.5)
_STAGE_STANDOFF = 0.028
SMOKE_APPROACH_DIST = 0.34 * BODY_LENGTH  # world units: fly-root to filter, "a hair" from the mouth


def _approach(source_xy, from_xy, standoff=_STAGE_STANDOFF):
    source = np.asarray(source_xy, dtype=np.float64)
    frm = np.asarray(from_xy, dtype=np.float64)
    d = source - frm
    n = np.linalg.norm(d)
    if n < 1e-6:
        d = np.array([1.0, 0.0])
        n = 1.0
    stand = source - (d / n) * standoff
    heading = math.atan2(d[1], d[0])
    return stand, heading


def _smoke_face_stand(scene: "AddictionScene3D", standoff_world: float = SMOKE_APPROACH_DIST):
    """Env-space (stand_xy, heading) for the fly to stand exactly in line
    with the cigarette's own long axis, facing down it toward the filter
    (and, beyond it, the glowing tip) -- rather than approaching from
    whatever direction it happened to walk from. Must be called after
    `scene.set_sources(...)` so `scene.ids.filter_world`/`tip_world` are
    up to date.
    """
    filt = scene.ids.filter_world[:2]
    tip = scene.ids.tip_world[:2]
    axis = filt - tip  # points from tip toward filter ("inward")
    n = np.linalg.norm(axis)
    axis_n = axis / n if n > 1e-9 else np.array([1.0, 0.0])
    stand_world = filt + axis_n * standoff_world  # a hair further inward than the filter
    heading = math.atan2(-axis_n[1], -axis_n[0])  # face back outward, down the cigarette's axis
    return world_to_env(stand_world), heading


def _staged_segments(scene: "AddictionScene3D"):
    smoke_stand, smoke_heading = _smoke_face_stand(scene)
    reels_stand, reels_heading = _approach(_STAGE_REELS_XY, smoke_stand)
    segs = [
        dict(kind="walk", t0=0.0, t1=2.0, frm=_STAGE_START_XY, to=smoke_stand),
        dict(kind="smoke", t0=2.0, t1=6.0, pos=smoke_stand, heading=smoke_heading),
        dict(kind="walk", t0=6.0, t1=8.0, frm=smoke_stand, to=reels_stand),
        dict(kind="reels", t0=8.0, t1=12.0, pos=reels_stand, heading=reels_heading,
             jackpots=[9.3, 10.7]),
    ]
    return segs


def _state_at(segs, t: float):
    t = float(np.clip(t, 0.0, segs[-1]["t1"]))
    seg = segs[-1]
    for s in segs:
        if s["t0"] <= t <= s["t1"]:
            seg = s
            break
    span = max(1e-6, seg["t1"] - seg["t0"])
    frac = float(np.clip((t - seg["t0"]) / span, 0.0, 1.0))
    if seg["kind"] == "walk":
        frm = np.asarray(seg["frm"], dtype=np.float64)
        to = np.asarray(seg["to"], dtype=np.float64)
        xy = frm + (to - frm) * frac
        d = to - frm
        heading = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-9 else 0.0
        at = None
    else:
        xy = np.asarray(seg["pos"], dtype=np.float64)
        heading = seg["heading"]
        at = "smoke" if seg["kind"] == "smoke" else "reels"

    hunger = float(np.clip(0.4 + 0.15 * (t / segs[-1]["t1"]), 0.0, 1.0))
    if t < 2.0:
        nicotine = 0.0
    elif t < 6.0:
        nicotine = float(np.clip((t - 2.0) / 4.0, 0.0, 1.0)) * 0.9
    else:
        nicotine = 0.9 * math.exp(-0.15 * (t - 6.0))
    tolerance = float(np.clip(0.05 + 0.08 * max(0.0, t - 2.0), 0.0, 0.6))

    jackpot = False
    if seg["kind"] == "reels":
        for jt in seg.get("jackpots", []):
            if jt <= t < jt + 0.05:
                jackpot = True
    return dict(x=xy[0], y=xy[1], heading=heading, at=at, hunger=hunger,
               nicotine=nicotine, tolerance=tolerance, jackpot=jackpot, t=t)


def render_staged_demo(out: str = "renders/staged_demo.mp4", fps: int = 30,
                       scene: Optional["AddictionScene3D"] = None):
    """~12s hand-authored clip: fly walks to the cigarette, smokes ~4s,
    walks to the phone, scrolls ~4s."""
    import imageio

    if scene is None:
        scene = AddictionScene3D()
    scene.set_sources(_STAGE_FOOD_XY, _STAGE_SMOKE_XY, _STAGE_REELS_XY)
    segs = _staged_segments(scene)
    total_t = segs[-1]["t1"]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer = imageio.get_writer(out, fps=fps, codec="libx264",
                                quality=None, bitrate=None, macro_block_size=1,
                                output_params=["-pix_fmt", "yuv420p", "-crf", "20"])
    n = int(total_t * fps)
    try:
        for i in range(n + 1):
            t = i / fps
            st = _state_at(segs, t)
            scene.set_state(**st)
            writer.append_data(scene.render(camera="auto"))
    finally:
        writer.close()
    return {"out": out, "n_frames": n + 1, "duration_s": total_t}


def render_staged_stills(out_dir: str = "renders", scene: Optional["AddictionScene3D"] = None):
    """Renders staged_smoking.png / staged_reels.png / staged_overview.png
    by stepping the same staged trajectory used by `render_staged_demo` at
    fine resolution (so smoke particles / scroll phase are populated
    realistically) and grabbing a frame at chosen moments.
    """
    if scene is None:
        scene = AddictionScene3D()
    scene.set_sources(_STAGE_FOOD_XY, _STAGE_SMOKE_XY, _STAGE_REELS_XY)
    segs = _staged_segments(scene)

    targets = {
        round(2.0 + 0.85, 3): ("staged_smoking.png", "closeup"),  # just past exhale trigger
        1.0: ("staged_overview.png", "top"),
        9.5: ("staged_reels.png", "closeup"),                     # mid-swipe + fresh jackpot
    }
    fine_fps = 60
    total_t = max(targets.keys()) + 0.02
    n = int(total_t * fine_fps)
    os.makedirs(out_dir, exist_ok=True)
    saved = {}
    for i in range(n + 1):
        t = i / fine_fps
        st = _state_at(segs, t)
        scene.set_state(**st)
        for target_t, (fname, cam) in targets.items():
            if fname in saved:
                continue
            if abs(t - target_t) <= 0.5 / fine_fps:
                img = scene.render(camera=cam)
                path = os.path.join(out_dir, fname)
                Image.fromarray(img).save(path)
                saved[fname] = path
    return saved


# --------------------------------------------------------------------------
# Staged "skip food" demo: a fully choreographed (non-policy) trajectory
# where the fly's route runs right by the food, it slows to consider the
# droplet and walks on without eating, then smokes a cigarette and scrolls
# reels on a phone. Unlike `render_staged_demo` (whose per-segment heading
# is defined independently per activity and can jump at a beat boundary),
# every interval here carries an explicit (heading0 -> heading1) pair that
# `_skip_food_state_at` eases continuously, so reorientation always either
# (a) happens while the fly is walking, with heading held exactly equal to
# the walk's own direction of travel (so the body never faces away from
# where it's actually moving -- no "sliding"), or (b) happens while the
# fly is stationary, turning in place between two walks/activities whose
# directions differ. See `_skip_food_layout`/`_skip_food_segments` for how
# the two "no-turn-needed" cases (the walk up to the food, and the walk up
# to the cigarette) are arranged geometrically so they don't need (b).
# --------------------------------------------------------------------------

BODY_LEN_ENV = 1.0 / ARENA_BODY_LENGTHS  # env units per fly body-length

# start/food/cigarette are laid out exactly colinear through the arena
# centre: `cigarette_pose` always orients the prop so its filter faces
# *outward* along the (centre -> smoke_xy) bearing (see its docstring), so
# placing the walk's start point on the opposite side of centre from the
# cigarette makes the fly's whole approach heading equal that same outward
# bearing -- it arrives already facing the filter, no turn-in-place needed.
_SKIP_PATH_ANGLE = math.radians(-28.0)   # bearing, start -> centre -> cigarette
_SKIP_R_START = 0.36                     # env units, centre -> start
_SKIP_R_CIG = 0.34                       # env units, centre -> cigarette (== filter position)
_SKIP_FOOD_FRAC = 0.42                   # fraction of the start->cig walk where food sits
_SKIP_FOOD_OFFSET_BL = 1.5               # body lengths, food's perpendicular offset off that line
_SKIP_PASS_FRAC = 0.88                   # fraction of start->cig walk for the "walked past" waypoint
_SKIP_TURN_FRACTION = 0.35               # how much of the full look-at-food angle the snub glance uses
_SKIP_REELS_OFFSET_ANGLE = math.radians(105.0)  # phone bearing off the cigarette, relative to the path
_SKIP_REELS_DIST = 0.20                  # env units, cigarette -> phone ("a short walk")

_SKIP_HUNGER_T_MAX = 22.0   # seconds: hunger (already high) reaches 1.0 by here, then holds maxed
_SKIP_NIC_PEAK = 0.95
_SKIP_NIC_DECAY = 0.12      # per second, once smoking ends (nicotine "decays" through the reels beat)
_SKIP_TOL_BASE = 0.08
_SKIP_TOL_CLIMB = 0.10      # per second, while smoking ("tolerance starts climbing", then stays high)


def _skip_food_layout():
    center = np.array([0.5, 0.5])
    path_dir = np.array([math.cos(_SKIP_PATH_ANGLE), math.sin(_SKIP_PATH_ANGLE)])
    perp = np.array([-path_dir[1], path_dir[0]])
    start = center - path_dir * _SKIP_R_START
    cig_src = center + path_dir * _SKIP_R_CIG
    path_len = float(np.linalg.norm(cig_src - start))
    p1 = start + path_dir * (path_len * _SKIP_FOOD_FRAC)          # where the fly pauses to snub
    food_xy = p1 + perp * (_SKIP_FOOD_OFFSET_BL * BODY_LEN_ENV)   # the droplet itself, just off the path
    m_xy = start + path_dir * (path_len * _SKIP_PASS_FRAC)        # waypoint after walking past the food
    reels_angle = _SKIP_PATH_ANGLE + _SKIP_REELS_OFFSET_ANGLE
    reels_src = cig_src + np.array([math.cos(reels_angle), math.sin(reels_angle)]) * _SKIP_REELS_DIST
    return dict(start=start, cig_src=cig_src, p1=p1, food_xy=food_xy, m_xy=m_xy,
               reels_src=reels_src, path_heading=_SKIP_PATH_ANGLE)


def _skip_food_segments(scene: "AddictionScene3D"):
    """Builds the ordered list of choreographed intervals. Must be called
    after `scene.set_sources(...)` (uses `_smoke_face_stand`/`_approach`,
    which read the just-updated prop world frames off `scene.ids`).
    """
    L = _skip_food_layout()
    start, p1, m_xy, food_xy = L["start"], L["p1"], L["m_xy"], L["food_xy"]
    path_heading = L["path_heading"]

    smoke_stand, smoke_heading = _smoke_face_stand(scene)
    smoke_stand = np.asarray(smoke_stand, dtype=np.float64)

    reels_stand, reels_heading = _approach(L["reels_src"], smoke_stand)
    reels_stand = np.asarray(reels_stand, dtype=np.float64)

    food_heading = math.atan2(food_xy[1] - p1[1], food_xy[0] - p1[0])
    wiggle_delta = _wrap_angle(food_heading - path_heading) * _SKIP_TURN_FRACTION

    segs = []
    t = 0.0

    def add(dur, pos0, pos1, h0, h1, at, camera, **extra):
        nonlocal t
        segs.append(dict(t0=t, t1=t + dur, pos0=np.asarray(pos0, dtype=np.float64),
                         pos1=np.asarray(pos1, dtype=np.float64), heading0=h0, heading1=h1,
                         at=at, camera=camera, **extra))
        t += dur

    # 1: wide establishing shot -- arena, all three props, fly parked at the start.
    add(3.0, start, start, path_heading, path_heading, None, "top")
    # 2: walk toward the food (eased accel/decel; heading == direction of travel throughout).
    add(2.0, start, p1, path_heading, path_heading, None, "closeup")
    # slow beside the droplet, pause, glance toward it and back (no proboscis, at stays "snub").
    add(1.0, p1, p1, path_heading, path_heading, "snub", "closeup", wiggle=wiggle_delta)
    # walk on past the food, without eating.
    add(1.6, p1, m_xy, path_heading, path_heading, None, "closeup")
    # 3: continue to the cigarette (arrives already facing the filter -- see module note above).
    add(1.2, m_xy, smoke_stand, path_heading, path_heading, None, "closeup")
    add(0.6, smoke_stand, smoke_stand, path_heading, smoke_heading, None, "closeup")
    add(6.0, smoke_stand, smoke_stand, smoke_heading, smoke_heading, "smoke", "closeup")
    # 4: turn in place, then walk to the phone.
    add(0.8, smoke_stand, smoke_stand, smoke_heading, reels_heading, None, "closeup")
    add(3.0, smoke_stand, reels_stand, reels_heading, reels_heading, None, "closeup")
    # 5: scroll reels, over-the-shoulder framing, at least two jackpots.
    add(8.0, reels_stand, reels_stand, reels_heading, reels_heading, "reels", "closeup",
        jackpots=[t + 2.1, t + 5.6])
    # 6: cut to a final wide shot -- fly at the phone, untouched food back in frame.
    add(2.0, reels_stand, reels_stand, reels_heading, reels_heading, None, "top")

    return segs


def _skip_food_state_at(segs, t: float):
    """Returns (state_dict, camera_name) for time `t` -- `state_dict` is
    ready to pass as `scene.set_state(**state_dict)`.
    """
    t = float(np.clip(t, 0.0, segs[-1]["t1"]))
    seg = segs[-1]
    for s in segs:
        if s["t0"] <= t <= s["t1"]:
            seg = s
            break
    span = max(1e-6, seg["t1"] - seg["t0"])
    frac = float(np.clip((t - seg["t0"]) / span, 0.0, 1.0))
    ease = smoothstep(frac)

    xy = seg["pos0"] + (seg["pos1"] - seg["pos0"]) * ease
    dh = _wrap_angle(seg["heading1"] - seg["heading0"])
    heading = seg["heading0"] + dh * ease
    if "wiggle" in seg:
        heading += seg["wiggle"] * hump(frac)   # smooth glance toward the food and back
    heading = _wrap_angle(heading)

    at = seg["at"]

    hunger = float(np.clip(0.8 + 0.2 * (t / _SKIP_HUNGER_T_MAX), 0.0, 1.0))

    smoke_seg = next(s for s in segs if s["at"] == "smoke")
    t_smoke0, t_smoke1 = smoke_seg["t0"], smoke_seg["t1"]
    if t <= t_smoke0:
        nicotine = 0.0
    elif t <= t_smoke1:
        nicotine = _SKIP_NIC_PEAK * smoothstep((t - t_smoke0) / max(1e-6, t_smoke1 - t_smoke0))
    else:
        nicotine = _SKIP_NIC_PEAK * math.exp(-_SKIP_NIC_DECAY * (t - t_smoke1))

    if t <= t_smoke0:
        tolerance = _SKIP_TOL_BASE
    elif t <= t_smoke1:
        tolerance = _SKIP_TOL_BASE + _SKIP_TOL_CLIMB * (t - t_smoke0)
    else:
        tolerance = _SKIP_TOL_BASE + _SKIP_TOL_CLIMB * (t_smoke1 - t_smoke0)

    jackpot = any(jt <= t < jt + 0.05 for jt in seg.get("jackpots", []))

    state = dict(x=xy[0], y=xy[1], heading=heading, at=at, hunger=hunger,
                nicotine=nicotine, tolerance=tolerance, jackpot=jackpot, t=t)
    return state, seg["camera"]


_SKIP_FOOD_CAPTION_TITLE = "Skips food. Smokes. Scrolls reels."
_SKIP_FOOD_CAPTION_SUBTITLE = "staged animation — flybody MuJoCo model"


def render_staged_skip_food(out: str = "renders/staged_skip_food.mp4", fps: int = 30,
                            scene: Optional["AddictionScene3D"] = None):
    """~29s hand-choreographed clip: the fly's walk to the cigarette runs
    right by the food; it slows, pauses, glances at the droplet and walks
    on without eating -- then smokes and scrolls reels. Staged/scripted,
    not driven by the 2D env or any policy.
    """
    import imageio

    if scene is None:
        scene = AddictionScene3D()
    L = _skip_food_layout()
    scene.set_sources(L["food_xy"], L["cig_src"], L["reels_src"])
    segs = _skip_food_segments(scene)
    total_t = segs[-1]["t1"]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer = imageio.get_writer(out, fps=fps, codec="libx264",
                                quality=None, bitrate=None, macro_block_size=1,
                                output_params=["-pix_fmt", "yuv420p", "-crf", "20"])
    n = int(total_t * fps)
    try:
        for i in range(n + 1):
            t = i / fps
            st, camera = _skip_food_state_at(segs, t)
            scene.set_state(**st)
            frame = scene.render(camera=camera)
            frame = scene._draw_caption(frame, _SKIP_FOOD_CAPTION_TITLE,
                                        subtitle=_SKIP_FOOD_CAPTION_SUBTITLE)
            writer.append_data(frame)
    finally:
        writer.close()
    return {"out": out, "n_frames": n + 1, "duration_s": total_t}


def render_skip_food_stills(out_dir: str = "renders", scene: Optional["AddictionScene3D"] = None):
    """Renders staged_skip_food_snub.png / _smoke.png / _reels.png by
    stepping the same choreography used by `render_staged_skip_food` at
    fine resolution and grabbing a frame at each chosen moment.
    """
    if scene is None:
        scene = AddictionScene3D()
    L = _skip_food_layout()
    scene.set_sources(L["food_xy"], L["cig_src"], L["reels_src"])
    segs = _skip_food_segments(scene)

    snub_seg = next(s for s in segs if s["at"] == "snub")
    smoke_seg = next(s for s in segs if s["at"] == "smoke")
    reels_seg = next(s for s in segs if s["at"] == "reels")

    targets = {
        round(snub_seg["t0"] + 0.5 * (snub_seg["t1"] - snub_seg["t0"]), 3):
            ("staged_skip_food_snub.png", "closeup"),
        round(smoke_seg["t0"] + 0.85, 3):    # just past an exhale trigger
            ("staged_skip_food_smoke.png", "closeup"),
        round(reels_seg["t0"] + 2.15, 3):    # right around the first jackpot
            ("staged_skip_food_reels.png", "closeup"),
    }
    fine_fps = 60
    total_t = max(targets.keys()) + 0.02
    n = int(total_t * fine_fps)
    os.makedirs(out_dir, exist_ok=True)
    saved = {}
    for i in range(n + 1):
        t = i / fine_fps
        st, _ = _skip_food_state_at(segs, t)
        scene.set_state(**st)
        for target_t, (fname, cam) in targets.items():
            if fname in saved:
                continue
            if abs(t - target_t) <= 0.5 / fine_fps:
                img = scene.render(camera=cam)
                path = os.path.join(out_dir, fname)
                Image.fromarray(img).save(path)
                saved[fname] = path
    return saved


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",
                        choices=["stills", "demo", "episode", "traj", "skip_food", "all"],
                        default="all")
    parser.add_argument("--policy", default="greedy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="renders")
    parser.add_argument("--traj", default=None,
                        help="path to a flyrl.evaluate traj_seed*.npz (--mode traj)")
    parser.add_argument("--out", default=None,
                        help="output mp4 path (--mode traj; default <out-dir>/traj.mp4)")
    parser.add_argument("--title", default=None,
                        help="caption title drawn at the bottom of the frame (--mode traj)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera", default="auto",
                        choices=["auto", "orbit", "closeup", "top"])
    parser.add_argument("--frames-per-step", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="truncate the render after this many seconds of output video")
    args = parser.parse_args()

    if args.mode in ("stills", "all"):
        paths = render_staged_stills(out_dir=args.out_dir)
        print("stills:", paths)
    if args.mode in ("demo", "all"):
        info = render_staged_demo(out=os.path.join(args.out_dir, "staged_demo.mp4"))
        print("staged demo:", info)
    if args.mode in ("episode", "all"):
        info = render_episode(policy_name=args.policy, seed=args.seed,
                              out=os.path.join(args.out_dir, f"episode_{args.policy}.mp4"))
        print("episode:", info)
    if args.mode == "traj":
        assert args.traj, "--traj <path to traj_seed*.npz> is required for --mode traj"
        out = args.out or os.path.join(args.out_dir, "traj.mp4")
        info = render_trajectory(args.traj, out, fps=args.fps, camera=args.camera,
                                 frames_per_step=args.frames_per_step, title=args.title,
                                 max_seconds=args.max_seconds)
        print("trajectory:", info)
    if args.mode == "skip_food":
        stills = render_skip_food_stills(out_dir=args.out_dir)
        print("skip_food stills:", stills)
        info = render_staged_skip_food(out=os.path.join(args.out_dir, "staged_skip_food.mp4"),
                                       fps=args.fps)
        print("skip_food demo:", info)


if __name__ == "__main__":
    _main()
