import json
import math
import random
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(
    page_title="Directed Energy Engagement Digital Twin",
    page_icon="⚡",
    layout="wide",
)

# ============================================================
# Theme: black night + digital green + fire orange
# ============================================================

HUD_GREEN = "#7CFF22"
HUD_GREEN_BRIGHT = "#A5FF4D"
HUD_GREEN_DIM = "#3B7A1A"
HUD_ORANGE = "#FF7A18"
HUD_ORANGE_BRIGHT = "#FF9D2E"
HUD_ORANGE_DIM = "#A84300"
HUD_TEXT = "#E9FFE1"
HUD_MUTED = "#9DBA96"
HUD_RED = "#FF4D3D"
CURRENT_TARGET_BLUE = "#2F6BFF"
HUD_BG = "#020402"
HUD_GRID = "#214A19"

st.markdown(
    f"""
    <style>
    :root {{
        --hud-green: {HUD_GREEN};
        --hud-green-bright: {HUD_GREEN_BRIGHT};
        --hud-green-dim: {HUD_GREEN_DIM};
        --hud-orange: {HUD_ORANGE};
        --hud-orange-bright: {HUD_ORANGE_BRIGHT};
        --hud-orange-dim: {HUD_ORANGE_DIM};
        --hud-text: {HUD_TEXT};
        --hud-muted: {HUD_MUTED};
        --hud-red: {HUD_RED};
        --hud-bg: {HUD_BG};
    }}

    html, body, [class*="css"] {{
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 72% 8%, rgba(255,122,24,0.075), transparent 30%),
            radial-gradient(circle at 16% 2%, rgba(124,255,34,0.045), transparent 25%),
            linear-gradient(180deg, #010201 0%, #020502 55%, #010201 100%);
        color: var(--hud-text);
    }}

    .block-container {{
        max-width: 1600px;
        padding-top: 1.3rem;
        padding-bottom: 3rem;
    }}

    h1, h3 {{
        color: var(--hud-green-bright) !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        text-shadow: 0 0 8px rgba(124,255,34,0.18);
    }}

    h2 {{
        color: var(--hud-orange-bright) !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        text-shadow: 0 0 8px rgba(255,122,24,0.20);
    }}

    p, label, .stMarkdown, .stCaption {{
        color: var(--hud-text);
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #020502 0%, #061006 100%);
        border-right: 1px solid rgba(124,255,34,0.24);
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, rgba(6,17,6,0.96), rgba(2,7,2,0.96));
        border: 1px solid rgba(124,255,34,0.36);
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow:
            0 0 14px rgba(124,255,34,0.05),
            inset 0 0 18px rgba(124,255,34,0.02);
    }}

    div[data-testid="stMetricLabel"] {{
        color: var(--hud-muted);
    }}

    div[data-testid="stMetricValue"] {{
        color: var(--hud-orange-bright);
        text-shadow: 0 0 6px rgba(255,122,24,0.18);
    }}

    button[data-baseweb="tab"] {{
        color: var(--hud-muted);
        background: rgba(4,12,4,0.72);
        border: 1px solid rgba(124,255,34,0.16);
        border-radius: 8px 8px 0 0;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--hud-orange-bright);
        border-color: rgba(255,122,24,0.45);
        box-shadow: inset 0 -2px 0 var(--hud-orange);
    }}

    .stButton > button, .stDownloadButton > button {{
        color: #061006;
        background: linear-gradient(180deg, #FF9D2E 0%, #FF6A00 100%);
        border: 1px solid #FFB45C;
        font-weight: 700;
        border-radius: 8px;
        box-shadow: 0 0 12px rgba(255,122,24,0.14);
    }}

    hr {{
        border-color: rgba(124,255,34,0.18);
    }}

    footer {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update({
    "figure.facecolor": HUD_BG,
    "axes.facecolor": HUD_BG,
    "axes.edgecolor": HUD_GREEN_DIM,
    "axes.labelcolor": HUD_TEXT,
    "xtick.color": HUD_MUTED,
    "ytick.color": HUD_MUTED,
    "text.color": HUD_TEXT,
    "grid.color": HUD_GRID,
    "grid.alpha": 0.35,
    "axes.titlecolor": HUD_GREEN_BRIGHT,
})


# ============================================================
# Data models
# ============================================================

@dataclass
class Environment:
    range_km: float
    humidity_pct: float
    visibility_km: float
    turbulence: float
    wind_mps: float
    ambient_temp_c: float
    angstrom_exponent: float
    humidity_absorption_km_inv_at_100pct: float
    wind_pointing_sensitivity_urad_per_mps: float


@dataclass
class Target:
    target_type: str
    speed_mps: float
    velocity_angle_deg: float
    initial_altitude_m: float
    flight_path_angle_deg: float
    aspect_factor: float
    maneuver_factor: float
    characteristic_radius_m: float
    absorptivity: float
    areal_heat_capacity_kj_m2k: float
    thermal_loss_coeff_kw_m2k: float
    failure_delta_t_c: float
    hardness_multiplier: float


@dataclass
class SensorState:
    radar_quality: float
    eo_ir_quality: float
    data_latency_ms: float
    dropped_measurement_rate: float
    track_update_hz: float
    range_measurement_sigma_m: float
    bearing_measurement_sigma_mrad: float
    process_accel_sigma_mps2: float


@dataclass
class HELState:
    requested_optical_source_power_kw: float
    wall_plug_efficiency: float
    optics_efficiency: float
    commanded_dwell_time_s: float
    wavelength_um: float
    beam_quality_m2: float
    additional_half_angle_divergence_mrad: float
    initial_beam_diameter_m: float
    base_pointing_jitter_mrad: float
    beam_director_max_rate_mrad_s: float
    beam_director_servo_time_constant_s: float


@dataclass
class PlatformState:
    stored_energy_kwh: float
    storage_max_discharge_kw: float
    generator_power_kw: float
    cooling_capacity_kw: float
    coolant_temp_c: float
    thermal_limit_c: float
    thermal_capacitance_kj_per_c: float
    subsystem_health: float


# ============================================================
# Utility functions
# ============================================================

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def target_initial_position_m(
    env: Environment,
    tgt: Target,
) -> np.ndarray:
    """
    Initial 3-D target position relative to the directed-energy platform.

    env.range_km is interpreted as initial horizontal ground range.
    tgt.initial_altitude_m is the physical modeled altitude above the platform plane.
    """
    return np.array(
        [
            env.range_km * 1000.0,
            0.0,
            tgt.initial_altitude_m,
        ],
        dtype=float,
    )


def target_velocity_vector_mps(
    tgt: Target,
) -> np.ndarray:
    """
    True 3-D constant-velocity target vector.

    velocity_angle_deg:
      0 deg   = horizontal component directly closing
      90 deg  = horizontal crossing
      180 deg = horizontal component receding

    flight_path_angle_deg:
      positive = climbing
      zero     = level
      negative = descending
    """
    horizontal_angle = math.radians(
        tgt.velocity_angle_deg
    )
    gamma = math.radians(
        tgt.flight_path_angle_deg
    )

    horizontal_speed = (
        tgt.speed_mps * math.cos(gamma)
    )

    vx = (
        -horizontal_speed
        * math.cos(horizontal_angle)
    )
    vy = (
        horizontal_speed
        * math.sin(horizontal_angle)
    )
    vz = (
        tgt.speed_mps
        * math.sin(gamma)
    )

    return np.array(
        [vx, vy, vz],
        dtype=float,
    )


def target_velocity_components_mps(
    tgt: Target,
):
    """
    Backward-compatible horizontal decomposition used only by legacy diagnostics.
    """
    v = target_velocity_vector_mps(tgt)
    return -float(v[0]), float(v[1])


def instantaneous_los_axis_rates_mrad_s(
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
):
    """
    Exact azimuth and elevation LOS rates for 3-D Cartesian relative motion.
    """
    x, y, z = (
        float(position_m[0]),
        float(position_m[1]),
        float(position_m[2]),
    )
    vx, vy, vz = (
        float(velocity_mps[0]),
        float(velocity_mps[1]),
        float(velocity_mps[2]),
    )

    rho2 = max(
        x * x + y * y,
        1e-12,
    )
    rho = math.sqrt(rho2)
    r2 = max(
        rho2 + z * z,
        1.0,
    )

    az_rate_rad_s = (
        x * vy - y * vx
    ) / rho2

    rho_dot = (
        x * vx + y * vy
    ) / max(rho, 1e-9)

    el_rate_rad_s = (
        rho * vz
        - z * rho_dot
    ) / r2

    elevation_rad = math.atan2(
        z,
        rho,
    )

    # Physical LOS angular speed on the unit sphere.
    los_rate_rad_s = math.sqrt(
        (
            az_rate_rad_s
            * math.cos(elevation_rad)
        ) ** 2
        + el_rate_rad_s**2
    )

    return {
        "azimuth_rate_mrad_s": az_rate_rad_s * 1000.0,
        "elevation_rate_mrad_s": el_rate_rad_s * 1000.0,
        "magnitude_mrad_s": los_rate_rad_s * 1000.0,
        "azimuth_rad": math.atan2(y, x),
        "elevation_rad": elevation_rad,
    }


def instantaneous_los_rate_mrad_s(
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
) -> float:
    return abs(
        instantaneous_los_axis_rates_mrad_s(
            position_m,
            velocity_mps,
        )["magnitude_mrad_s"]
    )


def line_of_sight_rate_mrad_s(
    env: Environment,
    tgt: Target,
) -> float:
    return instantaneous_los_rate_mrad_s(
        target_initial_position_m(env, tgt),
        target_velocity_vector_mps(tgt),
    )


def engagement_geometry(
    env: Environment,
    tgt: Target,
    model_zone_range_km: float = 25.0,
    minimum_range_m: float = 100.0,
):
    """
    Constant-velocity 3-D engagement geometry.

    CPA, minimum range, model-zone exit, and engagement horizon are all computed
    from the true 3-D relative position and velocity vectors.
    """
    r0 = target_initial_position_m(
        env,
        tgt,
    )
    v = target_velocity_vector_mps(
        tgt
    )

    speed2 = float(v @ v)
    r0_mag = float(
        np.linalg.norm(r0)
    )
    rmax = max(
        model_zone_range_km * 1000.0,
        r0_mag,
    )
    rmin = max(
        minimum_range_m,
        1.0,
    )

    if speed2 <= 1e-12:
        return {
            "time_to_cpa_s": float("inf"),
            "cpa_range_m": r0_mag,
            "cpa_position_m": r0.copy(),
            "time_to_min_range_s": float("inf"),
            "time_to_zone_exit_s": float("inf"),
            "time_to_ground_impact_s": float("inf"),
            "engagement_horizon_s": float("inf"),
        }

    t_cpa_raw = (
        -float(r0 @ v)
        / speed2
    )
    t_cpa = max(
        0.0,
        t_cpa_raw,
    )
    r_cpa = (
        r0 + v * t_cpa
    )
    cpa_range_m = float(
        np.linalg.norm(r_cpa)
    )

    def sphere_intersection_times(radius_m):
        a = speed2
        b = 2.0 * float(r0 @ v)
        c = float(r0 @ r0) - radius_m**2

        disc = (
            b * b - 4.0 * a * c
        )
        if disc < 0.0:
            return []

        root = math.sqrt(disc)
        roots = [
            (-b - root) / (2.0 * a),
            (-b + root) / (2.0 * a),
        ]
        return sorted(
            t for t in roots
            if t >= 0.0
        )

    min_roots = sphere_intersection_times(
        rmin
    )
    zone_roots = sphere_intersection_times(
        rmax
    )

    time_to_min_range_s = (
        min_roots[0]
        if min_roots
        else float("inf")
    )

    time_to_zone_exit_s = (
        zone_roots[-1]
        if zone_roots
        else float("inf")
    )

    if float(v[2]) < -1e-12:
        time_to_ground_impact_s = max(
            0.0,
            -float(r0[2]) / float(v[2]),
        )
    else:
        time_to_ground_impact_s = float("inf")

    initially_closing = (
        float(r0 @ v) < 0.0
    )

    if initially_closing:
        engagement_horizon_s = t_cpa
        if math.isfinite(
            time_to_min_range_s
        ):
            engagement_horizon_s = min(
                engagement_horizon_s,
                time_to_min_range_s,
            )
    else:
        engagement_horizon_s = (
            time_to_zone_exit_s
        )

    if math.isfinite(
        time_to_ground_impact_s
    ):
        engagement_horizon_s = min(
            engagement_horizon_s,
            time_to_ground_impact_s,
        )

    return {
        "time_to_cpa_s": t_cpa,
        "cpa_range_m": cpa_range_m,
        "cpa_position_m": r_cpa,
        "time_to_min_range_s": time_to_min_range_s,
        "time_to_zone_exit_s": time_to_zone_exit_s,
        "time_to_ground_impact_s": time_to_ground_impact_s,
        "engagement_horizon_s": max(
            0.0,
            engagement_horizon_s,
        ),
    }


def available_engagement_time_s(
    env: Environment,
    tgt: Target,
) -> float:
    return engagement_geometry(
        env,
        tgt,
    )["engagement_horizon_s"]


def effective_dwell_time_s(
    env: Environment,
    tgt: Target,
    hel: HELState,
) -> float:
    return min(
        hel.commanded_dwell_time_s,
        available_engagement_time_s(
            env,
            tgt,
        ),
    )


# ============================================================
# Atmosphere
# ============================================================

def atmospheric_extinction(
    env: Environment,
    hel: HELState,
):
    """
    Low-order Beer-Lambert extinction model.

    Aerosol extinction is derived from meteorological visibility using the
    Koschmieder relation at 0.55 µm and spectrally scaled by an Ångström
    exponent. A simple Rayleigh term and a user-visible humidity absorption
    term are added. This is still not MODTRAN or a line-by-line atmosphere.
    """
    wavelength_um = max(hel.wavelength_um, 0.2)
    visibility_km = max(env.visibility_km, 0.2)

    aerosol_550_km_inv = 3.912 / visibility_km
    aerosol_km_inv = aerosol_550_km_inv * (
        0.55 / wavelength_um
    ) ** env.angstrom_exponent

    # Generic sea-level Rayleigh extinction approximation near visible/NIR.
    rayleigh_km_inv = 0.0116 * (
        0.55 / wavelength_um
    ) ** 4.0

    humidity_km_inv = (
        env.humidity_absorption_km_inv_at_100pct
        * clamp(env.humidity_pct / 100.0)
    )

    total_extinction_km_inv = (
        aerosol_km_inv
        + rayleigh_km_inv
        + humidity_km_inv
    )

    optical_depth = total_extinction_km_inv * env.range_km
    transmission = clamp(
        math.exp(-optical_depth),
        0.001,
        0.999,
    )

    return {
        "transmission": transmission,
        "aerosol_extinction_km_inv": aerosol_km_inv,
        "rayleigh_extinction_km_inv": rayleigh_km_inv,
        "humidity_extinction_km_inv": humidity_km_inv,
        "total_extinction_km_inv": total_extinction_km_inv,
        "optical_depth": optical_depth,
    }


# ============================================================
# Sensor fusion and state-estimation covariance
# ============================================================

def detection_probability(
    env: Environment,
    tgt: Target,
    sensors: SensorState,
) -> float:
    """
    Normalized sensor-performance abstraction.

    Detection remains phenomenological. The tracking stage below is upgraded
    to covariance propagation rather than a scalar-only heuristic.
    """
    range_factor = math.exp(-env.range_km / 18.0)
    maneuver_penalty = 1.0 - 0.20 * tgt.maneuver_factor

    radar = sensors.radar_quality * range_factor
    eo = sensors.eo_ir_quality * min(
        1.0,
        env.visibility_km / max(env.range_km, 0.5),
    )

    fused = 1.0 - (1.0 - radar) * (1.0 - eo)
    fused *= maneuver_penalty

    # Detection probability describes observability / sensing only.
    # Data-delivery dropout is handled separately in measurement availability.
    return clamp(fused)


def classification_confidence(
    p_detect: float,
    sensors: SensorState,
    tgt: Target,
) -> float:
    latency_penalty = math.exp(
        -sensors.data_latency_ms / 1800.0
    )
    aspect_penalty = 0.75 + 0.25 * tgt.aspect_factor
    return clamp(
        p_detect
        * latency_penalty
        * aspect_penalty
    )


def cv_transition(
    dt: float,
):
    """
    6-state constant-velocity transition for [x, y, z, vx, vy, vz].
    """
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    return np.block([
        [I3, dt * I3],
        [Z3, I3],
    ])


def cv_process_noise(
    dt: float,
    accel_sigma_mps2: float,
):
    """
    Isotropic 3-D piecewise-constant acceleration process-noise covariance.
    """
    q = max(
        accel_sigma_mps2,
        1e-6,
    ) ** 2

    I3 = np.eye(3)

    return q * np.block([
        [
            (dt**4 / 4.0) * I3,
            (dt**3 / 2.0) * I3,
        ],
        [
            (dt**3 / 2.0) * I3,
            (dt**2) * I3,
        ],
    ])


def los_basis_from_position(
    position_m: np.ndarray,
):
    """
    Return a right-handed 3-D LOS basis:
      u_r  = radial/slant-range direction
      u_az = local azimuth transverse direction
      u_el = local elevation transverse direction
    """
    p = np.asarray(
        position_m,
        dtype=float,
    )[:3]

    range_m = max(
        float(np.linalg.norm(p)),
        1.0,
    )
    u_r = p / range_m

    reference_up = np.array(
        [0.0, 0.0, 1.0],
        dtype=float,
    )

    u_az = np.cross(
        reference_up,
        u_r,
    )

    if (
        float(np.linalg.norm(u_az))
        < 1e-8
    ):
        reference_up = np.array(
            [0.0, 1.0, 0.0],
            dtype=float,
        )
        u_az = np.cross(
            reference_up,
            u_r,
        )

    u_az = u_az / max(
        float(np.linalg.norm(u_az)),
        1e-12,
    )

    u_el = np.cross(
        u_r,
        u_az,
    )
    u_el = u_el / max(
        float(np.linalg.norm(u_el)),
        1e-12,
    )

    return (
        u_r,
        u_az,
        u_el,
        range_m,
    )


def track_measurement_covariance(
    env: Environment,
    sensors: SensorState,
    position_m: np.ndarray | None = None,
    detection_probability_value: float = 1.0,
    apply_availability_weighting: bool = True,
):
    """
    Approximate 3-D Cartesian measurement covariance derived from:
      * range uncertainty,
      * azimuth angular uncertainty,
      * elevation angular uncertainty.

    The same generic bearing sigma is used for both angular axes.
    """
    if position_m is None:
        position_m = np.array(
            [
                env.range_km * 1000.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

    (
        u_r,
        u_az,
        u_el,
        range_m,
    ) = los_basis_from_position(
        position_m
    )

    combined_quality = clamp(
        0.55 * sensors.radar_quality
        + 0.45 * sensors.eo_ir_quality,
        0.10,
        1.0,
    )

    sigma_range = (
        sensors.range_measurement_sigma_m
        / combined_quality
    )

    sigma_bearing_rad = (
        sensors.bearing_measurement_sigma_mrad
        / 1000.0
        / combined_quality
    )

    horizontal_range_m = max(
        math.hypot(
            float(position_m[0]),
            float(position_m[1]),
        ),
        1e-6,
    )
    cos_elevation = clamp(
        horizontal_range_m / range_m,
        1e-6,
        1.0,
    )

    sigma_az_cross = max(
        range_m
        * cos_elevation
        * sigma_bearing_rad,
        0.1,
    )
    sigma_el_cross = max(
        range_m
        * sigma_bearing_rad,
        0.1,
    )

    raw_measurement_availability = clamp(
        detection_probability_value
        * (
            1.0
            - sensors.dropped_measurement_rate
        ),
        0.0,
        1.0,
    )

    if apply_availability_weighting:
        information_availability = clamp(
            raw_measurement_availability,
            0.02,
            1.0,
        )
    else:
        information_availability = 1.0

    R_los = np.diag([
        sigma_range**2
        / information_availability,
        sigma_az_cross**2
        / information_availability,
        sigma_el_cross**2
        / information_availability,
    ])

    B = np.column_stack(
        [
            u_r,
            u_az,
            u_el,
        ]
    )

    R_cartesian = (
        B @ R_los @ B.T
    )

    return (
        R_cartesian,
        sigma_range,
        sigma_az_cross,
        sigma_el_cross,
        raw_measurement_availability,
    )


def initialize_track_covariance(
    env: Environment,
    tgt: Target,
    sensors: SensorState,
):
    """
    Initialize a conservative fixed-Cartesian 6x6 covariance:
    [x, y, z, vx, vy, vz].
    """
    initial_position_m = (
        target_initial_position_m(
            env,
            tgt,
        )
    )

    (
        _,
        sigma_range,
        sigma_az_cross,
        sigma_el_cross,
        _,
    ) = track_measurement_covariance(
        env,
        sensors,
        initial_position_m,
        detection_probability_value=1.0,
    )

    (
        u_r,
        u_az,
        u_el,
        _,
    ) = los_basis_from_position(
        initial_position_m
    )

    B = np.column_stack(
        [
            u_r,
            u_az,
            u_el,
        ]
    )

    P_pos_los = np.diag([
        (2.0 * sigma_range) ** 2,
        (2.0 * sigma_az_cross) ** 2,
        (2.0 * sigma_el_cross) ** 2,
    ])

    P_pos = (
        B @ P_pos_los @ B.T
    )

    velocity_sigma_init = max(
        tgt.speed_mps * 0.15,
        5.0,
    )

    P_vel = (
        velocity_sigma_init**2
        * np.eye(3)
    )

    return np.block([
        [
            P_pos,
            np.zeros((3, 3)),
        ],
        [
            np.zeros((3, 3)),
            P_vel,
        ],
    ])


def kalman_covariance_step(
    P: np.ndarray,
    env: Environment,
    tgt: Target,
    sensors: SensorState,
    dt_s: float,
    measurement_update: bool,
    position_m: np.ndarray | None = None,
    detection_probability_value: float = 1.0,
    apply_availability_weighting: bool = True,
):
    """
    Sequential 3-D constant-velocity covariance propagation.
    """
    dt_s = max(
        float(dt_s),
        0.0,
    )

    maneuver_accel_sigma = (
        sensors.process_accel_sigma_mps2
        * (
            1.0
            + 2.5 * tgt.maneuver_factor
        )
    )

    F = cv_transition(
        dt_s
    )
    Q = cv_process_noise(
        dt_s,
        maneuver_accel_sigma,
    )

    P_pred = (
        F @ P @ F.T + Q
    )

    if measurement_update:
        R, _, _, _, _ = (
            track_measurement_covariance(
                env,
                sensors,
                position_m=position_m,
                detection_probability_value=detection_probability_value,
                apply_availability_weighting=apply_availability_weighting,
            )
        )

        H = np.block([
            [
                np.eye(3),
                np.zeros((3, 3)),
            ]
        ])

        I6 = np.eye(6)

        S = (
            H @ P_pred @ H.T + R
        )
        K = (
            P_pred
            @ H.T
            @ np.linalg.inv(S)
        )

        A = I6 - K @ H

        P_post = (
            A @ P_pred @ A.T
            + K @ R @ K.T
        )
    else:
        P_post = P_pred

    return P_post


def covariance_metrics(
    P_filter: np.ndarray,
    env: Environment,
    tgt: Target,
    sensors: SensorState,
    position_m: np.ndarray | None = None,
):
    """
    Project the 6-state Cartesian covariance into the instantaneous 3-D LOS frame.
    """
    if position_m is None:
        position_m = (
            target_initial_position_m(
                env,
                tgt,
            )
        )

    (
        u_r,
        u_az,
        u_el,
        range_m,
    ) = los_basis_from_position(
        position_m
    )

    maneuver_accel_sigma = (
        sensors.process_accel_sigma_mps2
        * (
            1.0
            + 2.5 * tgt.maneuver_factor
        )
    )

    latency_s = max(
        sensors.data_latency_ms
        / 1000.0,
        0.0,
    )

    if latency_s > 0.0:
        F_lat = cv_transition(
            latency_s
        )
        Q_lat = cv_process_noise(
            latency_s,
            maneuver_accel_sigma,
        )
        P_eval = (
            F_lat
            @ P_filter
            @ F_lat.T
            + Q_lat
        )
    else:
        P_eval = P_filter.copy()

    P_pos = P_eval[:3, :3]
    P_vel = P_eval[3:, 3:]

    def projected_sigma(
        covariance,
        axis,
    ):
        return math.sqrt(
            max(
                float(
                    axis.T
                    @ covariance
                    @ axis
                ),
                0.0,
            )
        )

    radial_sigma_m = projected_sigma(
        P_pos,
        u_r,
    )
    az_cross_sigma_m = projected_sigma(
        P_pos,
        u_az,
    )
    el_cross_sigma_m = projected_sigma(
        P_pos,
        u_el,
    )

    radial_velocity_sigma_mps = projected_sigma(
        P_vel,
        u_r,
    )
    az_velocity_sigma_mps = projected_sigma(
        P_vel,
        u_az,
    )
    el_velocity_sigma_mps = projected_sigma(
        P_vel,
        u_el,
    )

    horizontal_range_m = max(
        math.hypot(
            float(position_m[0]),
            float(position_m[1]),
        ),
        1e-6,
    )
    cos_elevation = clamp(
        horizontal_range_m / range_m,
        1e-6,
        1.0,
    )

    az_angular_sigma_mrad = (
        az_cross_sigma_m
        / max(
            range_m * cos_elevation,
            1e-6,
        )
        * 1000.0
    )
    el_angular_sigma_mrad = (
        el_cross_sigma_m
        / range_m
        * 1000.0
    )

    # Conservative scalar angular uncertainty for the low-order aimpoint model.
    angular_sigma_mrad = max(
        az_angular_sigma_mrad,
        el_angular_sigma_mrad,
    )

    target_angular_radius_mrad = (
        tgt.characteristic_radius_m
        / range_m
        * 1000.0
    )

    track_quality = clamp(
        math.exp(
            -0.5
            * (
                angular_sigma_mrad
                / 0.5
            ) ** 2
        )
    )

    return {
        "filter_covariance": P_filter,
        "latency_predicted_covariance": P_eval,
        "radial_sigma_m": radial_sigma_m,
        "cross_sigma_m": max(
            az_cross_sigma_m,
            el_cross_sigma_m,
        ),
        "az_cross_sigma_m": az_cross_sigma_m,
        "el_cross_sigma_m": el_cross_sigma_m,
        "radial_velocity_sigma_mps": radial_velocity_sigma_mps,
        "cross_velocity_sigma_mps": max(
            az_velocity_sigma_mps,
            el_velocity_sigma_mps,
        ),
        "az_velocity_sigma_mps": az_velocity_sigma_mps,
        "el_velocity_sigma_mps": el_velocity_sigma_mps,
        "angular_sigma_mrad": angular_sigma_mrad,
        "az_angular_sigma_mrad": az_angular_sigma_mrad,
        "el_angular_sigma_mrad": el_angular_sigma_mrad,
        "target_angular_radius_mrad": target_angular_radius_mrad,
        "track_quality": track_quality,
    }


def track_covariance_metrics(
    env: Environment,
    tgt: Target,
    sensors: SensorState,
):
    """
    Convenience 3-D snapshot wrapper retained for diagnostics.
    """
    position_m = (
        target_initial_position_m(
            env,
            tgt,
        )
    )

    P = initialize_track_covariance(
        env,
        tgt,
        sensors,
    )

    update_dt = (
        1.0
        / max(
            sensors.track_update_hz,
            0.1,
        )
    )

    slant_env = Environment(
        range_km=max(
            float(
                np.linalg.norm(
                    position_m
                )
            ) / 1000.0,
            0.001,
        ),
        humidity_pct=env.humidity_pct,
        visibility_km=env.visibility_km,
        turbulence=env.turbulence,
        wind_mps=env.wind_mps,
        ambient_temp_c=env.ambient_temp_c,
        angstrom_exponent=env.angstrom_exponent,
        humidity_absorption_km_inv_at_100pct=env.humidity_absorption_km_inv_at_100pct,
        wind_pointing_sensitivity_urad_per_mps=env.wind_pointing_sensitivity_urad_per_mps,
    )

    p_detect = detection_probability(
        slant_env,
        tgt,
        sensors,
    )

    for _ in range(12):
        P = kalman_covariance_step(
            P,
            slant_env,
            tgt,
            sensors,
            update_dt,
            measurement_update=True,
            position_m=position_m,
            detection_probability_value=p_detect,
        )

    return covariance_metrics(
        P,
        slant_env,
        tgt,
        sensors,
        position_m=position_m,
    )


# ============================================================
# Pointing and beam geometry
# ============================================================

def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (
        angle_rad + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def beam_director_state_step(
    commanded_los_angle_rad: float,
    director_angle_rad: float,
    hel: HELState,
    dt_s: float,
):
    """
    Propagate a generic first-order beam-director state with an angular-rate limit.

    The unsaturated first-order response is integrated analytically over dt:

        delta_theta_first_order
            = error * (1 - exp(-dt / tau))

    Then the physical angular motion over the interval is rate-limited:

        |delta_theta| <= omega_max * dt

    This avoids explicit-Euler instability when dt is large relative to the servo
    time constant. The residual servo tracking error is deterministic, not a 1σ
    random uncertainty.
    """
    tau_s = max(
        hel.beam_director_servo_time_constant_s,
        1e-6,
    )
    max_rate_rad_s = max(
        hel.beam_director_max_rate_mrad_s / 1000.0,
        1e-9,
    )
    dt_s = max(
        float(dt_s),
        0.0,
    )

    error_rad = wrap_angle_rad(
        commanded_los_angle_rad
        - director_angle_rad
    )

    # Exact first-order response for constant commanded LOS angle over the step.
    response_fraction = (
        1.0
        - math.exp(
            -dt_s / tau_s
        )
        if dt_s > 0.0
        else 0.0
    )

    desired_delta_rad = (
        error_rad
        * response_fraction
    )

    max_delta_rad = (
        max_rate_rad_s
        * dt_s
    )

    actual_delta_rad = float(
        np.clip(
            desired_delta_rad,
            -max_delta_rad,
            max_delta_rad,
        )
    )

    new_director_angle_rad = wrap_angle_rad(
        director_angle_rad
        + actual_delta_rad
    )

    residual_error_rad = wrap_angle_rad(
        commanded_los_angle_rad
        - new_director_angle_rad
    )

    actual_rate_rad_s = (
        actual_delta_rad / dt_s
        if dt_s > 1e-12
        else 0.0
    )

    desired_average_rate_rad_s = (
        desired_delta_rad / dt_s
        if dt_s > 1e-12
        else 0.0
    )

    rate_utilization = (
        abs(actual_rate_rad_s)
        / max_rate_rad_s
    )
    rate_demand_ratio = (
        abs(desired_average_rate_rad_s)
        / max_rate_rad_s
    )

    return {
        "director_angle_rad": new_director_angle_rad,
        "actual_rate_mrad_s": actual_rate_rad_s * 1000.0,
        "unsaturated_rate_mrad_s": desired_average_rate_rad_s * 1000.0,
        "servo_tracking_error_mrad": abs(residual_error_rad) * 1000.0,
        "rate_utilization": rate_utilization,
        "rate_demand_ratio": rate_demand_ratio,
        "rate_saturated": abs(desired_delta_rad) > max_delta_rad + 1e-15,
    }


def stochastic_pointing_sigma_mrad(
    env: Environment,
    hel: HELState,
    tracking_angular_sigma_mrad: float,
):
    """
    RSS combination of stochastic / uncertainty-like pointing contributors only.
    Deterministic beam-director lag is handled separately.
    """
    wind_jitter_mrad = (
        env.wind_mps
        * env.wind_pointing_sensitivity_urad_per_mps
        / 1000.0
    )

    turbulence_wander_mrad = (
        0.04 * env.turbulence
    )

    sigma_mrad = math.sqrt(
        hel.base_pointing_jitter_mrad**2
        + wind_jitter_mrad**2
        + turbulence_wander_mrad**2
        + tracking_angular_sigma_mrad**2
    )

    return {
        "sigma_mrad": sigma_mrad,
        "wind_mrad": wind_jitter_mrad,
        "turbulence_mrad": turbulence_wander_mrad,
    }


def effective_pointing_error_mrad(
    stochastic_sigma_mrad: float,
    servo_tracking_error_mrad: float,
) -> float:
    """
    Effective RMS-like pointing error used by the low-order engagement-footprint model.

    The servo term is a deterministic lag, so this quantity is deliberately not
    labeled as a statistical 1σ value.
    """
    return math.sqrt(
        max(stochastic_sigma_mrad, 0.0) ** 2
        + max(servo_tracking_error_mrad, 0.0) ** 2
    )


def pointing_jitter_mrad(
    env: Environment,
    hel: HELState,
    tracking_angular_sigma_mrad: float,
    los_rate_mrad_s: float = 0.0,
) -> float:
    """
    Legacy diagnostic helper. It returns stochastic pointing uncertainty only.
    The authoritative dynamic model uses beam_director_state_step() explicitly.
    """
    return stochastic_pointing_sigma_mrad(
        env,
        hel,
        tracking_angular_sigma_mrad,
    )["sigma_mrad"]


def beam_spot_geometry(
    env: Environment,
    hel: HELState,
    effective_pointing_mrad: float,
):
    """
    Low-order effective spot model.

    Diffraction uses a Gaussian-beam half-angle approximation:
        theta ~= M^2 * lambda / (pi * w0)

    It is RSS-combined with a user-specified additional half-angle spread and
    pointing / track uncertainty.
    """
    range_m = env.range_km * 1000.0
    wavelength_m = hel.wavelength_um * 1e-6
    w0_m = max(
        hel.initial_beam_diameter_m / 2.0,
        1e-4,
    )

    diffraction_half_angle_rad = (
        hel.beam_quality_m2
        * wavelength_m
        / (math.pi * w0_m)
    )

    additional_half_angle_rad = (
        hel.additional_half_angle_divergence_mrad
        / 1000.0
    )

    turbulence_spread_rad = (
        0.025 * env.turbulence
        / 1000.0
    )

    effective_divergence_rad = math.sqrt(
        diffraction_half_angle_rad**2
        + additional_half_angle_rad**2
        + turbulence_spread_rad**2
    )

    pointing_sigma_rad = (
        effective_pointing_mrad
        / 1000.0
    )

    divergence_radius_m = (
        effective_divergence_rad * range_m
    )
    pointing_radius_m = (
        pointing_sigma_rad * range_m
    )

    spot_radius_m = math.sqrt(
        w0_m**2
        + divergence_radius_m**2
        + pointing_radius_m**2
    )

    spot_area_m2 = max(
        math.pi * spot_radius_m**2,
        1e-8,
    )

    return {
        "diffraction_half_angle_mrad": (
            diffraction_half_angle_rad * 1000.0
        ),
        "effective_half_angle_divergence_mrad": (
            effective_divergence_rad * 1000.0
        ),
        "spot_radius_m": spot_radius_m,
        "spot_diameter_m": 2.0 * spot_radius_m,
        "spot_area_m2": spot_area_m2,
    }


def aimpoint_margin_index(
    env: Environment,
    tgt: Target,
    effective_pointing_mrad: float,
) -> float:
    """
    Dimensionless margin index based on angular target radius relative to
    combined 1-sigma pointing / track uncertainty. It is not a probability.
    """
    range_m = max(
        env.range_km * 1000.0,
        1.0,
    )
    allowable_mrad = max(
        tgt.characteristic_radius_m
        / range_m
        * 1000.0,
        0.005,
    )

    ratio = (
        effective_pointing_mrad
        / allowable_mrad
    )

    return clamp(
        math.exp(-0.5 * ratio**2)
    )


# ============================================================
# Power and platform thermal response
# ============================================================

def requested_electrical_input_kw(
    hel: HELState,
) -> float:
    return (
        hel.requested_optical_source_power_kw
        / max(
            hel.wall_plug_efficiency,
            1e-6,
        )
    )


def power_and_thermal_response(
    platform: PlatformState,
    hel: HELState,
    env: Environment,
    dwell_s: float,
):
    """
    Enforces:
      generator power
      storage discharge-power limit
      stored-energy limit
      wall-plug efficiency

    Cooling is an ambient-limited lumped model.
    """
    requested_electrical_kw = (
        requested_electrical_input_kw(hel)
    )

    energy_limited_storage_kw = (
        platform.stored_energy_kwh
        * 3600.0
        / max(dwell_s, 1e-6)
    )

    storage_available_kw = min(
        platform.storage_max_discharge_kw,
        energy_limited_storage_kw,
    )

    total_available_kw = (
        platform.generator_power_kw
        + storage_available_kw
    )

    actual_electrical_kw = min(
        requested_electrical_kw,
        total_available_kw,
    )

    actual_optical_kw = (
        actual_electrical_kw
        * hel.wall_plug_efficiency
    )

    generator_contribution_kw = min(
        platform.generator_power_kw,
        actual_electrical_kw,
    )

    storage_draw_kw = max(
        0.0,
        actual_electrical_kw
        - generator_contribution_kw,
    )

    storage_energy_used_kwh = (
        storage_draw_kw
        * dwell_s
        / 3600.0
    )

    energy_remaining_kwh = max(
        0.0,
        platform.stored_energy_kwh
        - storage_energy_used_kwh,
    )

    conversion_heat_kw = max(
        0.0,
        actual_electrical_kw
        - actual_optical_kw,
    )

    # Conservative assumption that optical-train inefficiency is absorbed
    # internally. This is intentionally visible in the model documentation.
    optical_train_heat_kw = max(
        0.0,
        actual_optical_kw
        * (1.0 - hel.optics_efficiency),
    )

    internal_heat_kw = (
        conversion_heat_kw
        + optical_train_heat_kw
    )

    temperature_headroom_c = max(
        platform.thermal_limit_c
        - env.ambient_temp_c,
        1.0,
    )

    cooling_effectiveness = clamp(
        (
            platform.coolant_temp_c
            - env.ambient_temp_c
        )
        / temperature_headroom_c
    )

    cooling_removed_kw = (
        platform.cooling_capacity_kw
        * cooling_effectiveness
    )

    net_heat_kw = (
        internal_heat_kw
        - cooling_removed_kw
    )

    delta_t_c = (
        net_heat_kw
        * dwell_s
        / max(
            platform.thermal_capacitance_kj_per_c,
            1e-6,
        )
    )

    new_temp_c = max(
        env.ambient_temp_c,
        platform.coolant_temp_c
        + delta_t_c,
    )

    thermal_margin = clamp(
        (
            platform.thermal_limit_c
            - new_temp_c
        )
        / max(
            platform.thermal_limit_c
            - env.ambient_temp_c,
            1e-6,
        )
    )

    energy_margin = clamp(
        energy_remaining_kwh
        / max(
            platform.stored_energy_kwh,
            1e-6,
        )
    )

    power_availability_ratio = clamp(
        total_available_kw
        / max(
            requested_electrical_kw,
            1e-6,
        )
    )

    return {
        "requested_electrical_kw": requested_electrical_kw,
        "actual_electrical_kw": actual_electrical_kw,
        "actual_optical_kw": actual_optical_kw,
        "generator_contribution_kw": generator_contribution_kw,
        "storage_available_kw": storage_available_kw,
        "storage_draw_kw": storage_draw_kw,
        "storage_energy_used_kwh": storage_energy_used_kwh,
        "energy_remaining_kwh": energy_remaining_kwh,
        "conversion_heat_kw": conversion_heat_kw,
        "optical_train_heat_kw": optical_train_heat_kw,
        "internal_heat_kw": internal_heat_kw,
        "cooling_removed_kw": cooling_removed_kw,
        "net_heat_kw": net_heat_kw,
        "new_temp_c": new_temp_c,
        "thermal_margin": thermal_margin,
        "energy_margin": energy_margin,
        "power_availability_ratio": power_availability_ratio,
    }


# ============================================================
# Target thermal response
# ============================================================

def target_thermal_response(
    average_irradiance_kw_m2: float,
    dwell_s: float,
    env: Environment,
    tgt: Target,
):
    """
    Generic lumped surface thermal-response model.

        C_A * dT/dt = alpha * I - h_A * (T - T_ambient)

    where:
      C_A = areal heat capacity [kJ/m²-K]
      I   = average incident irradiance [kW/m²]
      h_A = effective thermal-loss coefficient [kW/m²-K]

    The resulting effect index is the modeled temperature rise normalized
    against a user-visible synthetic failure-temperature rise. It is not a
    damage probability.
    """
    c_areal = max(
        tgt.areal_heat_capacity_kj_m2k,
        1e-6,
    )

    q_absorbed_kw_m2 = (
        clamp(tgt.absorptivity)
        * max(
            average_irradiance_kw_m2,
            0.0,
        )
    )

    h_loss = max(
        tgt.thermal_loss_coeff_kw_m2k,
        0.0,
    )

    if h_loss > 1e-9:
        thermal_time_constant_s = (
            c_areal / h_loss
        )
        steady_delta_t_c = (
            q_absorbed_kw_m2
            / h_loss
        )
        delta_t_c = (
            steady_delta_t_c
            * (
                1.0
                - math.exp(
                    -dwell_s
                    / thermal_time_constant_s
                )
            )
        )
    else:
        thermal_time_constant_s = float("inf")
        delta_t_c = (
            q_absorbed_kw_m2
            * dwell_s
            / c_areal
        )

    target_surface_temp_c = (
        env.ambient_temp_c
        + delta_t_c
    )

    failure_delta_t_effective_c = (
        tgt.failure_delta_t_c
        * max(
            tgt.hardness_multiplier,
            0.1,
        )
    )

    thermal_effect_index = clamp(
        delta_t_c
        / max(
            failure_delta_t_effective_c,
            1e-6,
        )
    )

    absorbed_exposure_kj_m2 = (
        q_absorbed_kw_m2
        * dwell_s
    )

    return {
        "absorbed_heat_flux_kw_m2": q_absorbed_kw_m2,
        "absorbed_exposure_kj_m2": absorbed_exposure_kj_m2,
        "thermal_time_constant_s": thermal_time_constant_s,
        "target_delta_t_c": delta_t_c,
        "target_surface_temp_c": target_surface_temp_c,
        "effective_failure_delta_t_c": failure_delta_t_effective_c,
        "thermal_effect_index": thermal_effect_index,
    }


# ============================================================
# Coupled engagement model
# ============================================================

def readiness_score(
    p_detect,
    class_conf,
    track_quality,
    thermal_effect_index,
    thermal_margin,
    energy_margin,
    power_availability_ratio,
    health,
):
    """
    Readiness score deliberately excludes the aimpoint-margin index.

    Pointing / tracking uncertainty already reduces delivered irradiance through the
    effective spot-size model. Keeping the aimpoint margin as a hard decision gate,
    rather than another weighted term, avoids double-counting the same degradation.
    """
    return clamp(
        0.10 * p_detect
        + 0.10 * class_conf
        + 0.18 * track_quality
        + 0.28 * thermal_effect_index
        + 0.11 * thermal_margin
        + 0.08 * energy_margin
        + 0.07 * power_availability_ratio
        + 0.08 * health
    )


def engagement_recommendation(
    score,
    track_quality,
    aim_margin_index,
    thermal_margin,
    energy_margin,
    power_availability_ratio,
    dwell_s,
    beam_director_rate_utilization=0.0,
):
    if dwell_s <= 0.1:
        return "HOLD: Insufficient engagement time"
    if power_availability_ratio < 0.60:
        return "HOLD: Power-limited state"
    if thermal_margin < 0.12:
        return "HOLD: Thermal constraint"
    if energy_margin < 0.10:
        return "HOLD: Energy reserve constraint"
    if track_quality < 0.45:
        return "TRACK: Improve state estimate"
    if beam_director_rate_utilization > 1.0:
        return "HOLD: Beam-director rate limit"
    if aim_margin_index < 0.20:
        return "HOLD: Aimpoint margin constraint"
    if score >= 0.75:
        return "ENGAGE / CONTINUE"
    if score >= 0.55:
        return "CAUTION: Marginal engagement"
    return "HOLD / REASSESS"


def simulate_static_snapshot(
    env,
    tgt,
    sensors,
    hel,
    platform,
    noise=True,
):
    if noise:
        env = Environment(
            range_km=max(
                0.1,
                random.gauss(
                    env.range_km,
                    0.04 * env.range_km,
                ),
            ),
            humidity_pct=clamp(
                random.gauss(
                    env.humidity_pct,
                    3.0,
                ),
                0.0,
                100.0,
            ),
            visibility_km=max(
                0.5,
                random.gauss(
                    env.visibility_km,
                    0.08 * env.visibility_km,
                ),
            ),
            turbulence=clamp(
                random.gauss(
                    env.turbulence,
                    0.05,
                )
            ),
            wind_mps=max(
                0.0,
                random.gauss(
                    env.wind_mps,
                    1.0,
                ),
            ),
            ambient_temp_c=random.gauss(
                env.ambient_temp_c,
                1.0,
            ),
            angstrom_exponent=max(
                0.0,
                random.gauss(
                    env.angstrom_exponent,
                    0.10,
                ),
            ),
            humidity_absorption_km_inv_at_100pct=max(
                0.0,
                random.gauss(
                    env.humidity_absorption_km_inv_at_100pct,
                    0.002,
                ),
            ),
            wind_pointing_sensitivity_urad_per_mps=(
                env.wind_pointing_sensitivity_urad_per_mps
            ),
        )

        tgt = Target(
            target_type=tgt.target_type,
            speed_mps=max(
                1.0,
                random.gauss(
                    tgt.speed_mps,
                    0.03 * tgt.speed_mps,
                ),
            ),
            velocity_angle_deg=clamp(
                random.gauss(
                    tgt.velocity_angle_deg,
                    3.0,
                ),
                0.0,
                180.0,
            ),
            initial_altitude_m=max(
                0.0,
                random.gauss(
                    tgt.initial_altitude_m,
                    max(25.0, 0.03 * max(tgt.initial_altitude_m, 1.0)),
                ),
            ),
            flight_path_angle_deg=clamp(
                random.gauss(
                    tgt.flight_path_angle_deg,
                    2.0,
                ),
                -60.0,
                45.0,
            ),
            aspect_factor=clamp(
                random.gauss(
                    tgt.aspect_factor,
                    0.04,
                ),
                0.2,
                1.0,
            ),
            maneuver_factor=clamp(
                random.gauss(
                    tgt.maneuver_factor,
                    0.05,
                )
            ),
            characteristic_radius_m=max(
                0.05,
                random.gauss(
                    tgt.characteristic_radius_m,
                    0.05 * tgt.characteristic_radius_m,
                ),
            ),
            absorptivity=clamp(
                random.gauss(
                    tgt.absorptivity,
                    0.04,
                ),
                0.05,
                0.95,
            ),
            areal_heat_capacity_kj_m2k=max(
                0.1,
                random.gauss(
                    tgt.areal_heat_capacity_kj_m2k,
                    0.08 * tgt.areal_heat_capacity_kj_m2k,
                ),
            ),
            thermal_loss_coeff_kw_m2k=max(
                0.0,
                random.gauss(
                    tgt.thermal_loss_coeff_kw_m2k,
                    0.10 * max(tgt.thermal_loss_coeff_kw_m2k, 0.01),
                ),
            ),
            failure_delta_t_c=max(
                10.0,
                random.gauss(
                    tgt.failure_delta_t_c,
                    0.08 * tgt.failure_delta_t_c,
                ),
            ),
            hardness_multiplier=max(
                0.1,
                random.gauss(
                    tgt.hardness_multiplier,
                    0.08 * tgt.hardness_multiplier,
                ),
            ),
        )

        sensors = SensorState(
            radar_quality=clamp(
                random.gauss(
                    sensors.radar_quality,
                    0.03,
                )
            ),
            eo_ir_quality=clamp(
                random.gauss(
                    sensors.eo_ir_quality,
                    0.03,
                )
            ),
            data_latency_ms=max(
                0.0,
                random.gauss(
                    sensors.data_latency_ms,
                    20.0,
                ),
            ),
            dropped_measurement_rate=clamp(
                random.gauss(
                    sensors.dropped_measurement_rate,
                    0.01,
                ),
                0.0,
                0.5,
            ),
            track_update_hz=max(
                0.5,
                random.gauss(
                    sensors.track_update_hz,
                    0.05 * sensors.track_update_hz,
                ),
            ),
            range_measurement_sigma_m=max(
                0.1,
                random.gauss(
                    sensors.range_measurement_sigma_m,
                    0.08 * sensors.range_measurement_sigma_m,
                ),
            ),
            bearing_measurement_sigma_mrad=max(
                0.001,
                random.gauss(
                    sensors.bearing_measurement_sigma_mrad,
                    0.08 * sensors.bearing_measurement_sigma_mrad,
                ),
            ),
            process_accel_sigma_mps2=max(
                0.01,
                random.gauss(
                    sensors.process_accel_sigma_mps2,
                    0.10 * sensors.process_accel_sigma_mps2,
                ),
            ),
        )

        hel = HELState(
            requested_optical_source_power_kw=max(
                1.0,
                random.gauss(
                    hel.requested_optical_source_power_kw,
                    0.02 * hel.requested_optical_source_power_kw,
                ),
            ),
            wall_plug_efficiency=clamp(
                random.gauss(
                    hel.wall_plug_efficiency,
                    0.02,
                ),
                0.05,
                0.90,
            ),
            optics_efficiency=clamp(
                random.gauss(
                    hel.optics_efficiency,
                    0.02,
                ),
                0.05,
                1.0,
            ),
            commanded_dwell_time_s=hel.commanded_dwell_time_s,
            wavelength_um=hel.wavelength_um,
            beam_quality_m2=max(
                1.0,
                random.gauss(
                    hel.beam_quality_m2,
                    0.04 * hel.beam_quality_m2,
                ),
            ),
            additional_half_angle_divergence_mrad=max(
                0.0,
                random.gauss(
                    hel.additional_half_angle_divergence_mrad,
                    0.08 * max(
                        hel.additional_half_angle_divergence_mrad,
                        0.005,
                    ),
                ),
            ),
            initial_beam_diameter_m=hel.initial_beam_diameter_m,
            base_pointing_jitter_mrad=max(
                0.0,
                random.gauss(
                    hel.base_pointing_jitter_mrad,
                    0.01,
                ),
            ),
            beam_director_max_rate_mrad_s=hel.beam_director_max_rate_mrad_s,
            beam_director_servo_time_constant_s=hel.beam_director_servo_time_constant_s,
        )

        platform = PlatformState(
            stored_energy_kwh=platform.stored_energy_kwh,
            storage_max_discharge_kw=max(
                0.0,
                random.gauss(
                    platform.storage_max_discharge_kw,
                    0.04 * max(
                        platform.storage_max_discharge_kw,
                        1.0,
                    ),
                ),
            ),
            generator_power_kw=max(
                0.0,
                random.gauss(
                    platform.generator_power_kw,
                    0.04 * max(
                        platform.generator_power_kw,
                        1.0,
                    ),
                ),
            ),
            cooling_capacity_kw=max(
                0.0,
                random.gauss(
                    platform.cooling_capacity_kw,
                    0.04 * max(
                        platform.cooling_capacity_kw,
                        1.0,
                    ),
                ),
            ),
            coolant_temp_c=platform.coolant_temp_c,
            thermal_limit_c=platform.thermal_limit_c,
            thermal_capacitance_kj_per_c=max(
                1.0,
                random.gauss(
                    platform.thermal_capacitance_kj_per_c,
                    0.05 * platform.thermal_capacitance_kj_per_c,
                ),
            ),
            subsystem_health=clamp(
                random.gauss(
                    platform.subsystem_health,
                    0.01,
                ),
                0.5,
                1.0,
            ),
        )

    p_detect = detection_probability(
        env,
        tgt,
        sensors,
    )

    class_conf = classification_confidence(
        p_detect,
        sensors,
        tgt,
    )

    track = track_covariance_metrics(
        env,
        tgt,
        sensors,
    )

    # Track quality is derived only from estimator covariance.
    # Detection already influences covariance through measurement availability,
    # while classification remains an independent decision-support variable.
    track_quality = track["track_quality"]

    dwell_s = effective_dwell_time_s(
        env,
        tgt,
        hel,
    )

    atmosphere = atmospheric_extinction(
        env,
        hel,
    )

    platform_state = power_and_thermal_response(
        platform,
        hel,
        env,
        dwell_s,
    )

    static_los_rate_mrad_s = line_of_sight_rate_mrad_s(
        env,
        tgt,
    )

    static_stochastic_pointing = stochastic_pointing_sigma_mrad(
        env,
        hel,
        track["angular_sigma_mrad"],
    )

    # Diagnostic snapshot approximation: initialize the director on the current LOS,
    # so no fictitious transient servo bias is injected into a non-time-stepped view.
    static_servo_error_mrad = 0.0
    static_rate_utilization = min(
        1.0,
        abs(static_los_rate_mrad_s)
        / max(hel.beam_director_max_rate_mrad_s, 1e-6),
    )

    effective_pointing_mrad = effective_pointing_error_mrad(
        static_stochastic_pointing["sigma_mrad"],
        static_servo_error_mrad,
    )

    beam = beam_spot_geometry(
        env,
        hel,
        effective_pointing_mrad,
    )

    target_optical_power_kw = (
        platform_state["actual_optical_kw"]
        * hel.optics_efficiency
        * atmosphere["transmission"]
    )

    average_irradiance_kw_m2 = (
        target_optical_power_kw
        / beam["spot_area_m2"]
    )

    aim_margin = aimpoint_margin_index(
        env,
        tgt,
        effective_pointing_mrad,
    )

    target_thermal = target_thermal_response(
        average_irradiance_kw_m2,
        dwell_s,
        env,
        tgt,
    )

    score = readiness_score(
        p_detect,
        class_conf,
        track_quality,
        target_thermal["thermal_effect_index"],
        platform_state["thermal_margin"],
        platform_state["energy_margin"],
        platform_state["power_availability_ratio"],
        platform.subsystem_health,
    )

    recommendation = engagement_recommendation(
        score,
        track_quality,
        aim_margin,
        platform_state["thermal_margin"],
        platform_state["energy_margin"],
        platform_state["power_availability_ratio"],
        dwell_s,
        beam_director_rate_utilization=static_rate_utilization,
    )

    return {
        "Detection Probability": p_detect,
        "Classification Confidence": class_conf,
        "Track Quality": track_quality,
        "Track Cross-Range 1σ (m)": track["cross_sigma_m"],
        "Track Radial 1σ (m)": track["radial_sigma_m"],
        "Track Angular 1σ (mrad)": track["angular_sigma_mrad"],
        "Target Angular Radius (mrad)": track["target_angular_radius_mrad"],
        "LOS Rate (mrad/s)": line_of_sight_rate_mrad_s(env, tgt),
        "Aimpoint Margin Index": aim_margin,
        "Atmospheric Transmission": atmosphere["transmission"],
        "Aerosol Extinction (1/km)": atmosphere["aerosol_extinction_km_inv"],
        "Rayleigh Extinction (1/km)": atmosphere["rayleigh_extinction_km_inv"],
        "Humidity Extinction (1/km)": atmosphere["humidity_extinction_km_inv"],
        "Optical Depth": atmosphere["optical_depth"],
        "Available Engagement Time (s)": available_engagement_time_s(env, tgt),
        "Time to CPA (s)": engagement_geometry(env, tgt)["time_to_cpa_s"],
        "CPA Range (m)": engagement_geometry(env, tgt)["cpa_range_m"],
        "Effective Dwell Time (s)": dwell_s,
        "Requested Optical Source Power (kW)": hel.requested_optical_source_power_kw,
        "Actual Optical Source Power (kW)": platform_state["actual_optical_kw"],
        "Requested Electrical Input (kW)": platform_state["requested_electrical_kw"],
        "Actual Electrical Input (kW)": platform_state["actual_electrical_kw"],
        "Power Availability Ratio": platform_state["power_availability_ratio"],
        "Generator Contribution (kW)": platform_state["generator_contribution_kw"],
        "Storage Draw (kW)": platform_state["storage_draw_kw"],
        "Target Optical Power (kW)": target_optical_power_kw,
        "Diffraction Half-Angle (mrad)": beam["diffraction_half_angle_mrad"],
        "Effective Beam Half-Angle (mrad)": beam["effective_half_angle_divergence_mrad"],
        "Effective Pointing Error (mrad)": effective_pointing_mrad,
        "Servo Tracking Error (mrad)": static_servo_error_mrad,
        "Beam Director Rate Utilization": static_rate_utilization,
        "Spot Diameter (m)": beam["spot_diameter_m"],
        "Average Irradiance (kW/m^2)": average_irradiance_kw_m2,
        "Absorbed Heat Flux (kW/m^2)": target_thermal["absorbed_heat_flux_kw_m2"],
        "Absorbed Exposure (kJ/m^2)": target_thermal["absorbed_exposure_kj_m2"],
        "Target ΔT (C)": target_thermal["target_delta_t_c"],
        "Target Surface Temp (C)": target_thermal["target_surface_temp_c"],
        "Target Thermal Time Constant (s)": target_thermal["thermal_time_constant_s"],
        "Estimated Thermal Effect Index": target_thermal["thermal_effect_index"],
        "Coolant Temp After Dwell (C)": platform_state["new_temp_c"],
        "Thermal Margin": platform_state["thermal_margin"],
        "Energy Margin": platform_state["energy_margin"],
        "Storage Energy Used (kWh)": platform_state["storage_energy_used_kwh"],
        "Internal Heat (kW)": platform_state["internal_heat_kw"],
        "Cooling Removed (kW)": platform_state["cooling_removed_kw"],
        "Net Heat Load (kW)": platform_state["net_heat_kw"],
        "Readiness Score": score,
        "Recommendation": recommendation,
    }


def engagement_timeline(
    env,
    tgt,
    sensors,
    hel,
    platform,
    steps=50,
):
    horizon = min(
        hel.commanded_dwell_time_s,
        available_engagement_time_s(env, tgt),
    )
    if not math.isfinite(horizon):
        horizon = hel.commanded_dwell_time_s

    dt_s = max(
        horizon / max(steps, 1),
        0.02,
    )

    timeline, _ = simulate_time_stepped_engagement(
        env,
        tgt,
        sensors,
        hel,
        platform,
        dt_s=dt_s,
    )

    if timeline.empty:
        return pd.DataFrame({
            "Time (s)": [0.0],
            "Range (km)": [env.range_km],
            "Estimated Thermal Effect Index": [0.0],
            "Target Surface Temp (C)": [env.ambient_temp_c],
            "Coolant Temp (C)": [platform.coolant_temp_c],
            "Stored Energy Remaining (kWh)": [platform.stored_energy_kwh],
            "Average Irradiance (kW/m^2)": [0.0],
        })

    return timeline



def simulate_time_stepped_engagement(
    env: Environment,
    tgt: Target,
    sensors: SensorState,
    hel: HELState,
    platform: PlatformState,
    dt_s: float = 0.10,
    stochastic_measurements: bool = False,
):
    """
    Authoritative true 3-D constant-velocity engagement engine.

    State geometry is propagated in x, y, z. Slant range, CPA, LOS azimuth/elevation,
    two-axis beam-director tracking, measurement covariance, atmosphere, beam footprint,
    power/thermal state, and target heating all use the same 3-D trajectory.
    """
    geometry = engagement_geometry(
        env,
        tgt,
    )
    horizon_s = geometry[
        "engagement_horizon_s"
    ]

    total_time_s = min(
        hel.commanded_dwell_time_s,
        horizon_s,
    )

    if not math.isfinite(
        total_time_s
    ):
        total_time_s = (
            hel.commanded_dwell_time_s
        )

    total_time_s = max(
        0.0,
        total_time_s,
    )

    r0 = target_initial_position_m(
        env,
        tgt,
    )
    velocity_mps = (
        target_velocity_vector_mps(
            tgt
        )
    )

    initial_slant_range_m = max(
        float(np.linalg.norm(r0)),
        1.0,
    )
    initial_los = (
        instantaneous_los_axis_rates_mrad_s(
            r0,
            velocity_mps,
        )
    )

    if total_time_s <= 0.0:
        slant_env = Environment(
            range_km=initial_slant_range_m / 1000.0,
            humidity_pct=env.humidity_pct,
            visibility_km=env.visibility_km,
            turbulence=env.turbulence,
            wind_mps=env.wind_mps,
            ambient_temp_c=env.ambient_temp_c,
            angstrom_exponent=env.angstrom_exponent,
            humidity_absorption_km_inv_at_100pct=env.humidity_absorption_km_inv_at_100pct,
            wind_pointing_sensitivity_urad_per_mps=env.wind_pointing_sensitivity_urad_per_mps,
        )
        atmosphere_zero = atmospheric_extinction(
            slant_env,
            hel,
        )

        rate_utilization = (
            initial_los["magnitude_mrad_s"]
            / max(
                hel.beam_director_max_rate_mrad_s,
                1e-6,
            )
        )

        return pd.DataFrame(), {
            "Detection Probability": 0.0,
            "Classification Confidence": 0.0,
            "Track Quality": 0.0,
            "Track Cross-Range 1σ (m)": float("nan"),
            "Track Azimuth Cross-Range 1σ (m)": float("nan"),
            "Track Elevation Cross-Range 1σ (m)": float("nan"),
            "Track Radial 1σ (m)": float("nan"),
            "Track Angular 1σ (mrad)": float("nan"),
            "Track Azimuth 1σ (mrad)": float("nan"),
            "Track Elevation 1σ (mrad)": float("nan"),
            "Target Angular Radius (mrad)": (
                tgt.characteristic_radius_m
                / initial_slant_range_m
                * 1000.0
            ),
            "LOS Rate (mrad/s)": initial_los["magnitude_mrad_s"],
            "Azimuth LOS Rate (mrad/s)": initial_los["azimuth_rate_mrad_s"],
            "Elevation LOS Rate (mrad/s)": initial_los["elevation_rate_mrad_s"],
            "Azimuth (deg)": math.degrees(initial_los["azimuth_rad"]),
            "Elevation Angle (deg)": math.degrees(initial_los["elevation_rad"]),
            "Altitude (m)": float(r0[2]),
            "Measurement Availability": 0.0,
            "Aimpoint Margin Index": 0.0,
            "Atmospheric Transmission": atmosphere_zero["transmission"],
            "Aerosol Extinction (1/km)": atmosphere_zero["aerosol_extinction_km_inv"],
            "Rayleigh Extinction (1/km)": atmosphere_zero["rayleigh_extinction_km_inv"],
            "Humidity Extinction (1/km)": atmosphere_zero["humidity_extinction_km_inv"],
            "Optical Depth": atmosphere_zero["optical_depth"],
            "Available Engagement Time (s)": 0.0,
            "Time to CPA (s)": geometry["time_to_cpa_s"],
            "Time to Ground Impact (s)": geometry["time_to_ground_impact_s"],
            "CPA Range (m)": geometry["cpa_range_m"],
            "Effective Dwell Time (s)": 0.0,
            "Requested Optical Source Power (kW)": hel.requested_optical_source_power_kw,
            "Actual Optical Source Power (kW)": 0.0,
            "Requested Electrical Input (kW)": requested_electrical_input_kw(hel),
            "Actual Electrical Input (kW)": 0.0,
            "Power Availability Ratio": 0.0,
            "Generator Contribution (kW)": 0.0,
            "Storage Draw (kW)": 0.0,
            "Target Optical Power (kW)": 0.0,
            "Diffraction Half-Angle (mrad)": 0.0,
            "Effective Beam Half-Angle (mrad)": 0.0,
            "Effective Pointing Error (mrad)": float("nan"),
            "Stochastic Pointing 1σ (mrad)": float("nan"),
            "Servo Tracking Error (mrad)": float("nan"),
            "Azimuth Servo Error (mrad)": float("nan"),
            "Elevation Servo Error (mrad)": float("nan"),
            "Beam Director Rate Utilization": min(rate_utilization, 1.0),
            "Beam Director Rate Demand Ratio": rate_utilization,
            "Beam Director Rate Saturated": rate_utilization > 1.0,
            "Spot Diameter (m)": float("nan"),
            "Average Irradiance (kW/m^2)": 0.0,
            "Absorbed Heat Flux (kW/m^2)": 0.0,
            "Absorbed Exposure (kJ/m^2)": 0.0,
            "Target ΔT (C)": 0.0,
            "Target Surface Temp (C)": env.ambient_temp_c,
            "Target Thermal Time Constant (s)": float("inf"),
            "Estimated Thermal Effect Index": 0.0,
            "Coolant Temp After Dwell (C)": platform.coolant_temp_c,
            "Thermal Margin": 1.0,
            "Energy Margin": 1.0,
            "Storage Energy Used (kWh)": 0.0,
            "Stored Energy Remaining (kWh)": platform.stored_energy_kwh,
            "Internal Heat (kW)": 0.0,
            "Cooling Removed (kW)": 0.0,
            "Net Heat Load (kW)": 0.0,
            "Readiness Score": 0.0,
            "Recommendation": "HOLD: Insufficient engagement time",
            "Final Range (km)": initial_slant_range_m / 1000.0,
            "X (km)": float(r0[0]) / 1000.0,
            "Y (km)": float(r0[1]) / 1000.0,
            "Z (km)": float(r0[2]) / 1000.0,
        }

    integration_dt = max(
        min(
            float(dt_s),
            1.0
            / max(
                sensors.track_update_hz,
                0.1,
            ),
        ),
        0.01,
    )

    steps = max(
        1,
        int(
            math.ceil(
                total_time_s
                / integration_dt
            )
        ),
    )

    dt_actual = (
        total_time_s / steps
    )

    target_temp_c = (
        env.ambient_temp_c
    )
    coolant_temp_c = (
        platform.coolant_temp_c
    )
    stored_energy_kwh = (
        platform.stored_energy_kwh
    )
    absorbed_exposure_kj_m2 = 0.0

    P_filter = initialize_track_covariance(
        env,
        tgt,
        sensors,
    )

    measurement_period_s = (
        1.0
        / max(
            sensors.track_update_hz,
            0.1,
        )
    )
    measurement_accumulator_s = (
        measurement_period_s
    )

    initial_az_rad = math.atan2(
        r0[1],
        r0[0],
    )
    initial_el_rad = math.atan2(
        r0[2],
        math.hypot(
            r0[0],
            r0[1],
        ),
    )

    director_azimuth_rad = (
        initial_az_rad
    )
    director_elevation_rad = (
        initial_el_rad
    )

    rows = []

    for k in range(steps):
        t_mid = (
            (k + 0.5)
            * dt_actual
        )
        elapsed_s = (
            (k + 1)
            * dt_actual
        )

        position_mid_m = (
            r0
            + velocity_mps * t_mid
        )
        position_end_m = (
            r0
            + velocity_mps * elapsed_s
        )

        range_mid_m = max(
            float(
                np.linalg.norm(
                    position_mid_m
                )
            ),
            1.0,
        )
        range_end_m = max(
            float(
                np.linalg.norm(
                    position_end_m
                )
            ),
            1.0,
        )

        range_mid_km = (
            range_mid_m / 1000.0
        )
        range_end_km = (
            range_end_m / 1000.0
        )

        mid_env = Environment(
            range_km=range_mid_km,
            humidity_pct=env.humidity_pct,
            visibility_km=env.visibility_km,
            turbulence=env.turbulence,
            wind_mps=env.wind_mps,
            ambient_temp_c=env.ambient_temp_c,
            angstrom_exponent=env.angstrom_exponent,
            humidity_absorption_km_inv_at_100pct=env.humidity_absorption_km_inv_at_100pct,
            wind_pointing_sensitivity_urad_per_mps=env.wind_pointing_sensitivity_urad_per_mps,
        )

        end_env = Environment(
            range_km=range_end_km,
            humidity_pct=env.humidity_pct,
            visibility_km=env.visibility_km,
            turbulence=env.turbulence,
            wind_mps=env.wind_mps,
            ambient_temp_c=env.ambient_temp_c,
            angstrom_exponent=env.angstrom_exponent,
            humidity_absorption_km_inv_at_100pct=env.humidity_absorption_km_inv_at_100pct,
            wind_pointing_sensitivity_urad_per_mps=env.wind_pointing_sensitivity_urad_per_mps,
        )

        step_platform = PlatformState(
            stored_energy_kwh=stored_energy_kwh,
            storage_max_discharge_kw=platform.storage_max_discharge_kw,
            generator_power_kw=platform.generator_power_kw,
            cooling_capacity_kw=platform.cooling_capacity_kw,
            coolant_temp_c=coolant_temp_c,
            thermal_limit_c=platform.thermal_limit_c,
            thermal_capacitance_kj_per_c=platform.thermal_capacitance_kj_per_c,
            subsystem_health=platform.subsystem_health,
        )

        p_detect = detection_probability(
            end_env,
            tgt,
            sensors,
        )
        class_conf = classification_confidence(
            p_detect,
            sensors,
            tgt,
        )

        measurement_availability = clamp(
            p_detect
            * (
                1.0
                - sensors.dropped_measurement_rate
            ),
            0.0,
            1.0,
        )

        measurement_accumulator_s += (
            dt_actual
        )
        scheduled_measurement = (
            measurement_accumulator_s
            + 1e-12
            >= measurement_period_s
        )

        if scheduled_measurement:
            if stochastic_measurements:
                do_measurement_update = (
                    random.random()
                    < measurement_availability
                )
            else:
                do_measurement_update = (
                    measurement_availability
                    > 0.02
                )
        else:
            do_measurement_update = False

        P_filter = kalman_covariance_step(
            P_filter,
            end_env,
            tgt,
            sensors,
            dt_actual,
            measurement_update=do_measurement_update,
            position_m=position_end_m,
            detection_probability_value=p_detect,
            apply_availability_weighting=not stochastic_measurements,
        )

        if scheduled_measurement:
            measurement_accumulator_s -= (
                measurement_period_s
            )

        track = covariance_metrics(
            P_filter,
            end_env,
            tgt,
            sensors,
            position_m=position_end_m,
        )
        track_quality = (
            track["track_quality"]
        )

        atmosphere = atmospheric_extinction(
            mid_env,
            hel,
        )

        step_power = power_and_thermal_response(
            step_platform,
            hel,
            mid_env,
            dt_actual,
        )

        los = instantaneous_los_axis_rates_mrad_s(
            position_end_m,
            velocity_mps,
        )

        commanded_azimuth_rad = (
            los["azimuth_rad"]
        )
        commanded_elevation_rad = (
            los["elevation_rad"]
        )

        az_state = beam_director_state_step(
            commanded_azimuth_rad,
            director_azimuth_rad,
            hel,
            dt_actual,
        )
        el_state = beam_director_state_step(
            commanded_elevation_rad,
            director_elevation_rad,
            hel,
            dt_actual,
        )

        director_azimuth_rad = (
            az_state["director_angle_rad"]
        )
        director_elevation_rad = (
            el_state["director_angle_rad"]
        )

        stochastic_pointing = (
            stochastic_pointing_sigma_mrad(
                mid_env,
                hel,
                track["angular_sigma_mrad"],
            )
        )

        az_servo_error_mrad = (
            az_state[
                "servo_tracking_error_mrad"
            ]
            * abs(
                math.cos(
                    commanded_elevation_rad
                )
            )
        )
        el_servo_error_mrad = (
            el_state[
                "servo_tracking_error_mrad"
            ]
        )

        servo_tracking_error_mrad = math.sqrt(
            az_servo_error_mrad**2
            + el_servo_error_mrad**2
        )

        effective_pointing_mrad = (
            effective_pointing_error_mrad(
                stochastic_pointing[
                    "sigma_mrad"
                ],
                servo_tracking_error_mrad,
            )
        )

        beam = beam_spot_geometry(
            mid_env,
            hel,
            effective_pointing_mrad,
        )

        target_optical_power_kw = (
            step_power[
                "actual_optical_kw"
            ]
            * hel.optics_efficiency
            * atmosphere[
                "transmission"
            ]
        )

        average_irradiance_kw_m2 = (
            target_optical_power_kw
            / beam["spot_area_m2"]
        )

        aim_margin = aimpoint_margin_index(
            end_env,
            tgt,
            effective_pointing_mrad,
        )

        c_areal = max(
            tgt.areal_heat_capacity_kj_m2k,
            1e-6,
        )
        absorbed_flux_kw_m2 = (
            clamp(
                tgt.absorptivity
            )
            * max(
                average_irradiance_kw_m2,
                0.0,
            )
        )

        h_loss = max(
            tgt.thermal_loss_coeff_kw_m2k,
            0.0,
        )

        heat_loss_kw_m2 = (
            h_loss
            * max(
                target_temp_c
                - env.ambient_temp_c,
                0.0,
            )
        )

        target_delta_t_step_c = (
            (
                absorbed_flux_kw_m2
                - heat_loss_kw_m2
            )
            * dt_actual
            / c_areal
        )

        target_temp_c = max(
            env.ambient_temp_c,
            target_temp_c
            + target_delta_t_step_c,
        )

        absorbed_exposure_kj_m2 += (
            absorbed_flux_kw_m2
            * dt_actual
        )

        target_delta_t_total_c = (
            target_temp_c
            - env.ambient_temp_c
        )

        effective_failure_delta_t_c = (
            tgt.failure_delta_t_c
            * max(
                tgt.hardness_multiplier,
                0.1,
            )
        )

        thermal_effect_index = clamp(
            target_delta_t_total_c
            / max(
                effective_failure_delta_t_c,
                1e-6,
            )
        )

        thermal_time_constant_s = (
            c_areal / h_loss
            if h_loss > 1e-12
            else float("inf")
        )

        coolant_temp_c = (
            step_power["new_temp_c"]
        )
        stored_energy_kwh = (
            step_power[
                "energy_remaining_kwh"
            ]
        )

        energy_margin = clamp(
            stored_energy_kwh
            / max(
                platform.stored_energy_kwh,
                1e-6,
            )
        )

        score = readiness_score(
            p_detect,
            class_conf,
            track_quality,
            thermal_effect_index,
            step_power[
                "thermal_margin"
            ],
            energy_margin,
            step_power[
                "power_availability_ratio"
            ],
            platform.subsystem_health,
        )

        director_rate_utilization = max(
            az_state[
                "rate_utilization"
            ],
            el_state[
                "rate_utilization"
            ],
        )
        director_rate_demand_ratio = max(
            az_state[
                "rate_demand_ratio"
            ],
            el_state[
                "rate_demand_ratio"
            ],
        )
        director_rate_saturated = (
            az_state["rate_saturated"]
            or el_state["rate_saturated"]
        )

        recommendation = engagement_recommendation(
            score,
            track_quality,
            aim_margin,
            step_power["thermal_margin"],
            energy_margin,
            step_power[
                "power_availability_ratio"
            ],
            elapsed_s,
            beam_director_rate_utilization=director_rate_demand_ratio,
        )

        state = {
            "Time (s)": elapsed_s,
            "Range (km)": range_end_km,
            "Physics Evaluation Range (km)": range_mid_km,
            "Final Range (km)": range_end_km,
            "X (km)": float(position_end_m[0]) / 1000.0,
            "Y (km)": float(position_end_m[1]) / 1000.0,
            "Z (km)": float(position_end_m[2]) / 1000.0,
            "Altitude (m)": float(position_end_m[2]),
            "Azimuth (deg)": math.degrees(commanded_azimuth_rad),
            "Elevation Angle (deg)": math.degrees(commanded_elevation_rad),
            "Detection Probability": p_detect,
            "Classification Confidence": class_conf,
            "Measurement Availability": measurement_availability,
            "Track Quality": track_quality,
            "Track Cross-Range 1σ (m)": track["cross_sigma_m"],
            "Track Azimuth Cross-Range 1σ (m)": track["az_cross_sigma_m"],
            "Track Elevation Cross-Range 1σ (m)": track["el_cross_sigma_m"],
            "Track Radial 1σ (m)": track["radial_sigma_m"],
            "Track Angular 1σ (mrad)": track["angular_sigma_mrad"],
            "Track Azimuth 1σ (mrad)": track["az_angular_sigma_mrad"],
            "Track Elevation 1σ (mrad)": track["el_angular_sigma_mrad"],
            "Target Angular Radius (mrad)": track["target_angular_radius_mrad"],
            "LOS Rate (mrad/s)": los["magnitude_mrad_s"],
            "Azimuth LOS Rate (mrad/s)": los["azimuth_rate_mrad_s"],
            "Elevation LOS Rate (mrad/s)": los["elevation_rate_mrad_s"],
            "Aimpoint Margin Index": aim_margin,
            "Atmospheric Transmission": atmosphere["transmission"],
            "Aerosol Extinction (1/km)": atmosphere["aerosol_extinction_km_inv"],
            "Rayleigh Extinction (1/km)": atmosphere["rayleigh_extinction_km_inv"],
            "Humidity Extinction (1/km)": atmosphere["humidity_extinction_km_inv"],
            "Optical Depth": atmosphere["optical_depth"],
            "Available Engagement Time (s)": geometry["engagement_horizon_s"],
            "Time to CPA (s)": geometry["time_to_cpa_s"],
            "Time to Ground Impact (s)": geometry["time_to_ground_impact_s"],
            "CPA Range (m)": geometry["cpa_range_m"],
            "Effective Dwell Time (s)": elapsed_s,
            "Requested Optical Source Power (kW)": hel.requested_optical_source_power_kw,
            "Actual Optical Source Power (kW)": step_power["actual_optical_kw"],
            "Requested Electrical Input (kW)": step_power["requested_electrical_kw"],
            "Actual Electrical Input (kW)": step_power["actual_electrical_kw"],
            "Power Availability Ratio": step_power["power_availability_ratio"],
            "Generator Contribution (kW)": step_power["generator_contribution_kw"],
            "Storage Draw (kW)": step_power["storage_draw_kw"],
            "Target Optical Power (kW)": target_optical_power_kw,
            "Diffraction Half-Angle (mrad)": beam["diffraction_half_angle_mrad"],
            "Effective Beam Half-Angle (mrad)": beam["effective_half_angle_divergence_mrad"],
            "Effective Pointing Error (mrad)": effective_pointing_mrad,
            "Stochastic Pointing 1σ (mrad)": stochastic_pointing["sigma_mrad"],
            "Servo Tracking Error (mrad)": servo_tracking_error_mrad,
            "Azimuth Servo Error (mrad)": az_servo_error_mrad,
            "Elevation Servo Error (mrad)": el_servo_error_mrad,
            "Beam Director Rate Utilization": director_rate_utilization,
            "Beam Director Rate Demand Ratio": director_rate_demand_ratio,
            "Beam Director Rate Saturated": director_rate_saturated,
            "Spot Diameter (m)": beam["spot_diameter_m"],
            "Average Irradiance (kW/m^2)": average_irradiance_kw_m2,
            "Absorbed Heat Flux (kW/m^2)": absorbed_flux_kw_m2,
            "Absorbed Exposure (kJ/m^2)": absorbed_exposure_kj_m2,
            "Target ΔT (C)": target_delta_t_total_c,
            "Target Surface Temp (C)": target_temp_c,
            "Target Thermal Time Constant (s)": thermal_time_constant_s,
            "Estimated Thermal Effect Index": thermal_effect_index,
            "Coolant Temp After Dwell (C)": coolant_temp_c,
            "Coolant Temp (C)": coolant_temp_c,
            "Thermal Margin": step_power["thermal_margin"],
            "Energy Margin": energy_margin,
            "Storage Energy Used (kWh)": (
                platform.stored_energy_kwh
                - stored_energy_kwh
            ),
            "Stored Energy Remaining (kWh)": stored_energy_kwh,
            "Internal Heat (kW)": step_power["internal_heat_kw"],
            "Cooling Removed (kW)": step_power["cooling_removed_kw"],
            "Net Heat Load (kW)": step_power["net_heat_kw"],
            "Readiness Score": score,
            "Recommendation": recommendation,
        }

        rows.append(state)

    timeline = pd.DataFrame(rows)
    final_state = dict(
        rows[-1]
    )

    return (
        timeline,
        final_state,
    )


def perturb_scenario(
    env: Environment,
    tgt: Target,
    sensors: SensorState,
    hel: HELState,
    platform: PlatformState,
):
    """Generate one generic uncertainty realization for dynamic Monte Carlo."""
    p_env = Environment(
        range_km=max(
            0.1,
            random.gauss(
                env.range_km,
                0.04 * env.range_km,
            ),
        ),
        humidity_pct=clamp(
            random.gauss(
                env.humidity_pct,
                3.0,
            ),
            0.0,
            100.0,
        ),
        visibility_km=max(
            0.5,
            random.gauss(
                env.visibility_km,
                0.08 * env.visibility_km,
            ),
        ),
        turbulence=clamp(
            random.gauss(
                env.turbulence,
                0.05,
            )
        ),
        wind_mps=max(
            0.0,
            random.gauss(
                env.wind_mps,
                1.0,
            ),
        ),
        ambient_temp_c=random.gauss(
            env.ambient_temp_c,
            1.0,
        ),
        angstrom_exponent=max(
            0.0,
            random.gauss(
                env.angstrom_exponent,
                0.10,
            ),
        ),
        humidity_absorption_km_inv_at_100pct=max(
            0.0,
            random.gauss(
                env.humidity_absorption_km_inv_at_100pct,
                0.002,
            ),
        ),
        wind_pointing_sensitivity_urad_per_mps=env.wind_pointing_sensitivity_urad_per_mps,
    )

    p_tgt = Target(
        target_type=tgt.target_type,
        speed_mps=max(
            1.0,
            random.gauss(
                tgt.speed_mps,
                0.03 * tgt.speed_mps,
            ),
        ),
        velocity_angle_deg=clamp(
            random.gauss(
                tgt.velocity_angle_deg,
                3.0,
            ),
            0.0,
            180.0,
        ),
        initial_altitude_m=max(
            0.0,
            random.gauss(
                tgt.initial_altitude_m,
                max(
                    25.0,
                    0.03 * max(
                        tgt.initial_altitude_m,
                        1.0,
                    ),
                ),
            ),
        ),
        flight_path_angle_deg=clamp(
            random.gauss(
                tgt.flight_path_angle_deg,
                2.0,
            ),
            -60.0,
            45.0,
        ),
        aspect_factor=clamp(
            random.gauss(
                tgt.aspect_factor,
                0.04,
            ),
            0.2,
            1.0,
        ),
        maneuver_factor=clamp(
            random.gauss(
                tgt.maneuver_factor,
                0.05,
            )
        ),
        characteristic_radius_m=max(
            0.05,
            random.gauss(
                tgt.characteristic_radius_m,
                0.05 * tgt.characteristic_radius_m,
            ),
        ),
        absorptivity=clamp(
            random.gauss(
                tgt.absorptivity,
                0.04,
            ),
            0.05,
            0.95,
        ),
        areal_heat_capacity_kj_m2k=max(
            0.1,
            random.gauss(
                tgt.areal_heat_capacity_kj_m2k,
                0.08 * tgt.areal_heat_capacity_kj_m2k,
            ),
        ),
        thermal_loss_coeff_kw_m2k=max(
            0.0,
            random.gauss(
                tgt.thermal_loss_coeff_kw_m2k,
                0.10 * max(
                    tgt.thermal_loss_coeff_kw_m2k,
                    0.01,
                ),
            ),
        ),
        failure_delta_t_c=max(
            10.0,
            random.gauss(
                tgt.failure_delta_t_c,
                0.08 * tgt.failure_delta_t_c,
            ),
        ),
        hardness_multiplier=max(
            0.1,
            random.gauss(
                tgt.hardness_multiplier,
                0.08 * tgt.hardness_multiplier,
            ),
        ),
    )

    p_sensors = SensorState(
        radar_quality=clamp(
            random.gauss(
                sensors.radar_quality,
                0.03,
            )
        ),
        eo_ir_quality=clamp(
            random.gauss(
                sensors.eo_ir_quality,
                0.03,
            )
        ),
        data_latency_ms=max(
            0.0,
            random.gauss(
                sensors.data_latency_ms,
                20.0,
            ),
        ),
        dropped_measurement_rate=clamp(
            random.gauss(
                sensors.dropped_measurement_rate,
                0.01,
            ),
            0.0,
            0.5,
        ),
        track_update_hz=max(
            0.5,
            random.gauss(
                sensors.track_update_hz,
                0.05 * sensors.track_update_hz,
            ),
        ),
        range_measurement_sigma_m=max(
            0.1,
            random.gauss(
                sensors.range_measurement_sigma_m,
                0.08 * sensors.range_measurement_sigma_m,
            ),
        ),
        bearing_measurement_sigma_mrad=max(
            0.001,
            random.gauss(
                sensors.bearing_measurement_sigma_mrad,
                0.08 * sensors.bearing_measurement_sigma_mrad,
            ),
        ),
        process_accel_sigma_mps2=max(
            0.01,
            random.gauss(
                sensors.process_accel_sigma_mps2,
                0.10 * sensors.process_accel_sigma_mps2,
            ),
        ),
    )

    p_hel = HELState(
        requested_optical_source_power_kw=max(
            1.0,
            random.gauss(
                hel.requested_optical_source_power_kw,
                0.02 * hel.requested_optical_source_power_kw,
            ),
        ),
        wall_plug_efficiency=clamp(
            random.gauss(
                hel.wall_plug_efficiency,
                0.02,
            ),
            0.05,
            0.90,
        ),
        optics_efficiency=clamp(
            random.gauss(
                hel.optics_efficiency,
                0.02,
            ),
            0.05,
            1.0,
        ),
        commanded_dwell_time_s=hel.commanded_dwell_time_s,
        wavelength_um=hel.wavelength_um,
        beam_quality_m2=max(
            1.0,
            random.gauss(
                hel.beam_quality_m2,
                0.04 * hel.beam_quality_m2,
            ),
        ),
        additional_half_angle_divergence_mrad=max(
            0.0,
            random.gauss(
                hel.additional_half_angle_divergence_mrad,
                0.08 * max(
                    hel.additional_half_angle_divergence_mrad,
                    0.005,
                ),
            ),
        ),
        initial_beam_diameter_m=hel.initial_beam_diameter_m,
        base_pointing_jitter_mrad=max(
            0.0,
            random.gauss(
                hel.base_pointing_jitter_mrad,
                0.01,
            ),
        ),
        beam_director_max_rate_mrad_s=hel.beam_director_max_rate_mrad_s,
        beam_director_servo_time_constant_s=hel.beam_director_servo_time_constant_s,
    )

    p_platform = PlatformState(
        stored_energy_kwh=max(
            0.1,
            random.gauss(
                platform.stored_energy_kwh,
                0.02 * platform.stored_energy_kwh,
            ),
        ),
        storage_max_discharge_kw=max(
            0.0,
            random.gauss(
                platform.storage_max_discharge_kw,
                0.04 * max(
                    platform.storage_max_discharge_kw,
                    1.0,
                ),
            ),
        ),
        generator_power_kw=max(
            0.0,
            random.gauss(
                platform.generator_power_kw,
                0.04 * max(
                    platform.generator_power_kw,
                    1.0,
                ),
            ),
        ),
        cooling_capacity_kw=max(
            0.0,
            random.gauss(
                platform.cooling_capacity_kw,
                0.04 * max(
                    platform.cooling_capacity_kw,
                    1.0,
                ),
            ),
        ),
        coolant_temp_c=platform.coolant_temp_c,
        thermal_limit_c=platform.thermal_limit_c,
        thermal_capacitance_kj_per_c=max(
            1.0,
            random.gauss(
                platform.thermal_capacitance_kj_per_c,
                0.05 * platform.thermal_capacitance_kj_per_c,
            ),
        ),
        subsystem_health=clamp(
            random.gauss(
                platform.subsystem_health,
                0.01,
            ),
            0.5,
            1.0,
        ),
    )

    return (
        p_env,
        p_tgt,
        p_sensors,
        p_hel,
        p_platform,
    )


def simulate_dynamic_monte_carlo_run(
    env,
    tgt,
    sensors,
    hel,
    platform,
):
    (
        p_env,
        p_tgt,
        p_sensors,
        p_hel,
        p_platform,
    ) = perturb_scenario(
        env,
        tgt,
        sensors,
        hel,
        platform,
    )

    _, final_state = simulate_time_stepped_engagement(
        p_env,
        p_tgt,
        p_sensors,
        p_hel,
        p_platform,
        dt_s=0.10,
        stochastic_measurements=True,
    )

    return final_state



def build_3d_digital_twin_figure(
    timeline: pd.DataFrame,
    env: Environment,
    tgt: Target,
    result: dict,
    selected_index: int,
    camera_preset: str = "Isometric",
    show_target_path: bool = True,
    show_cpa: bool = True,
    show_uncertainty: bool = True,
    show_beam_footprint: bool = True,
    show_engagement_zone: bool = True,
    show_event_markers: bool = True,
    enable_animation: bool = True,
):
    """
    Interactive 3-D visualization layer for the authoritative engagement timeline.

    The viewer renders the authoritative true 3-D constant-velocity engagement state.
    The same x/y/z trajectory drives slant range, tracking, pointing, propagation,
    thermal calculations, event markers, and visualization.
    """
    if timeline is None or timeline.empty:
        return go.Figure()

    selected_index = int(
        max(
            0,
            min(
                int(selected_index),
                len(timeline) - 1,
            ),
        )
    )

    times_s = timeline["Time (s)"].to_numpy(dtype=float)

    positions_m = np.column_stack(
        [
            timeline["X (km)"].to_numpy(dtype=float) * 1000.0,
            timeline["Y (km)"].to_numpy(dtype=float) * 1000.0,
            timeline["Z (km)"].to_numpy(dtype=float) * 1000.0,
        ]
    )

    x_km = positions_m[:, 0] / 1000.0
    y_km = positions_m[:, 1] / 1000.0
    z_km = positions_m[:, 2] / 1000.0

    t_sel = float(times_s[selected_index])
    p_sel = positions_m[selected_index]

    x_sel_km = float(p_sel[0] / 1000.0)
    y_sel_km = float(p_sel[1] / 1000.0)
    z_sel_km = float(p_sel[2] / 1000.0)

    # UI-only auto-fit. The authoritative x/y/z trajectory remains unchanged.
    all_x = np.concatenate([x_km, np.array([0.0])])
    all_y = np.concatenate([y_km, np.array([0.0])])
    all_z = np.concatenate([z_km, np.array([0.0])])

    def _axis_range(values, min_span=0.8, pad_fraction=0.12):
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        span = max(vmax - vmin, min_span)
        pad = max(span * pad_fraction, 0.15)
        center = 0.5 * (vmin + vmax)
        half = 0.5 * span + pad
        return [center - half, center + half]

    x_range = _axis_range(all_x)
    y_range = _axis_range(all_y)
    z_range = _axis_range(all_z, min_span=0.5, pad_fraction=0.15)

    x_span = max(x_range[1] - x_range[0], 1e-6)
    y_span = max(y_range[1] - y_range[0], 1e-6)
    z_span = max(z_range[1] - z_range[0], 1e-6)
    max_span = max(x_span, y_span, z_span)

    aspect_ratio = dict(
        x=max(x_span / max_span, 0.28),
        y=max(y_span / max_span, 0.28),
        z=max(z_span / max_span, 0.22),
    )

    geometry = engagement_geometry(env, tgt)
    t_cpa = geometry["time_to_cpa_s"]
    p_cpa = geometry["cpa_position_m"]

    if math.isfinite(t_cpa):
        cpa_x_km = float(p_cpa[0] / 1000.0)
        cpa_y_km = float(p_cpa[1] / 1000.0)
        cpa_z_km = float(p_cpa[2] / 1000.0)
    else:
        cpa_x_km = float("nan")
        cpa_y_km = float("nan")
        cpa_z_km = float("nan")

    theta = np.linspace(
        0.0,
        2.0 * math.pi,
        181,
    )
    zone_radius_km = max(
        25.0,
        env.range_km,
    )
    zone_x = zone_radius_km * np.cos(theta)
    zone_y = zone_radius_km * np.sin(theta)
    zone_z = np.zeros_like(zone_x)

    plane_extent_km = max(
        zone_radius_km,
        np.nanmax(np.abs(x_km)) + 2.0,
        np.nanmax(np.abs(y_km)) + 2.0,
    )
    grid = np.linspace(
        -plane_extent_km,
        plane_extent_km,
        2,
    )
    xx, yy = np.meshgrid(grid, grid)
    zz = np.zeros_like(xx)

    # Dynamic target color maps low effect to green, medium to fire orange,
    # and high effect to warning red.
    effect_value = float(
        timeline.iloc[selected_index]["Estimated Thermal Effect Index"]
    )
    if effect_value < 0.40:
        target_color = HUD_GREEN_BRIGHT
    elif effect_value < 0.75:
        target_color = HUD_ORANGE_BRIGHT
    else:
        target_color = HUD_RED

    camera_map = {
        "Isometric": dict(
            eye=dict(x=1.10, y=1.10, z=0.78),
            center=dict(x=0.0, y=0.0, z=-0.02),
        ),
        "Top-Down": dict(
            eye=dict(x=0.0, y=0.0, z=1.85),
            center=dict(x=0.0, y=0.0, z=0.0),
        ),
        "Tactical": dict(
            eye=dict(x=1.35, y=0.42, z=0.52),
            center=dict(x=0.0, y=0.0, z=-0.03),
        ),
        "Beam Sight": dict(
            eye=dict(x=-0.12, y=-0.10, z=0.24),
            center=dict(x=0.0, y=0.0, z=0.0),
        ),
        "Target Chase": dict(
            eye=dict(x=0.42, y=1.45, z=0.45),
            center=dict(x=0.0, y=0.0, z=-0.02),
        ),
    }
    camera = camera_map.get(
        camera_preset,
        camera_map["Isometric"],
    )

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            surfacecolor=np.zeros_like(zz),
            colorscale=[
                [0.0, "#020402"],
                [1.0, "#061006"],
            ],
            opacity=0.92,
            showscale=False,
            hoverinfo="skip",
            name="Reference Plane",
        )
    )

    if show_engagement_zone:
        fig.add_trace(
            go.Scatter3d(
                x=zone_x,
                y=zone_y,
                z=zone_z,
                mode="lines",
                line=dict(
                    color=HUD_GREEN_DIM,
                    width=2,
                    dash="dot",
                ),
                name="Model Zone",
                hoverinfo="skip",
            )
        )

    if show_target_path:
        fig.add_trace(
            go.Scatter3d(
                x=x_km,
                y=y_km,
                z=z_km,
                mode="lines",
                line=dict(
                    color=HUD_ORANGE,
                    width=6,
                ),
                name="Target Path",
                customdata=np.column_stack(
                    [
                        times_s,
                        timeline["Range (km)"].to_numpy(dtype=float),
                    ]
                ),
                hovertemplate=(
                    "<b>Target path</b><br>"
                    "Time: %{customdata[0]:.2f} s<br>"
                    "Range: %{customdata[1]:.2f} km"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            marker=dict(
                size=13,
                color=HUD_GREEN_BRIGHT,
                symbol="diamond",
                line=dict(
                    color="#E9FFE1",
                    width=1,
                ),
            ),
            text=["DE Platform"],
            textposition="top center",
            name="DE Platform",
            hovertemplate=(
                "<b>Directed-energy platform</b>"
                "<extra></extra>"
            ),
        )
    )
    platform_trace_index = len(fig.data) - 1

    # Beam / LOS.
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, x_sel_km],
            y=[0.0, y_sel_km],
            z=[0.0, z_sel_km],
            mode="lines",
            line=dict(
                color=HUD_GREEN_BRIGHT,
                width=max(
                    4,
                    min(
                        12,
                        4
                        + 2
                        * float(
                            timeline.iloc[selected_index]["Spot Diameter (m)"]
                        ),
                    ),
                ),
            ),
            name="Beam / LOS",
            hovertemplate=(
                "Beam / LOS"
                "<extra></extra>"
            ),
        )
    )
    beam_trace_index = len(fig.data) - 1

    # Current target with rich telemetry.
    row = timeline.iloc[selected_index]

    # Thermal-state halo: identity stays blue while the halo communicates
    # Estimated Thermal Effect Index using green / orange / red.
    fig.add_trace(
        go.Scatter3d(
            x=[x_sel_km],
            y=[y_sel_km],
            z=[z_sel_km],
            mode="markers",
            marker=dict(
                size=22,
                color="rgba(0,0,0,0)",
                symbol="circle",
                line=dict(
                    color=target_color,
                    width=5,
                ),
            ),
            name="Target State",
            hovertemplate=(
                "<b>Target thermal state</b><br>"
                f"Estimated thermal effect index: {effect_value:.1%}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    target_halo_trace_index = len(fig.data) - 1

    fig.add_trace(
        go.Scatter3d(
            x=[x_sel_km],
            y=[y_sel_km],
            z=[z_sel_km],
            mode="markers+text",
            marker=dict(
                size=13,
                color=CURRENT_TARGET_BLUE,
                symbol="circle",
                line=dict(
                    color="#DCE7FF",
                    width=1.5,
                ),
            ),
            text=["Current Target"],
            textposition="top center",
            textfont=dict(
                color=CURRENT_TARGET_BLUE,
            ),
            name="Current Target",
            customdata=[[
                t_sel,
                float(row["Range (km)"]),
                float(row["LOS Rate (mrad/s)"]),
                float(row["Altitude (m)"]),
                float(row["Elevation Angle (deg)"]),
                float(row["Track Angular 1σ (mrad)"]),
                float(row["Atmospheric Transmission"]),
                float(row["Average Irradiance (kW/m^2)"]),
                float(row["Target ΔT (C)"]),
                float(row["Estimated Thermal Effect Index"]),
                float(row["Power Availability Ratio"]),
                float(row["Stored Energy Remaining (kWh)"]),
            ]],
            hovertemplate=(
                "<b>Current target</b><br>"
                "Time: %{customdata[0]:.2f} s<br>"
                "Range: %{customdata[1]:.2f} km<br>"
                "LOS rate: %{customdata[2]:.3f} mrad/s<br>"
                "Altitude: %{customdata[3]:.0f} m<br>"
                "Elevation: %{customdata[4]:.2f}°<br>"
                "Track 1σ: %{customdata[5]:.3f} mrad<br>"
                "Atmospheric transmission: %{customdata[6]:.1%}<br>"
                "Avg. irradiance: %{customdata[7]:.2f} kW/m²<br>"
                "Target ΔT: %{customdata[8]:.1f} °C<br>"
                "Thermal effect index: %{customdata[9]:.1%}<br>"
                "Power availability: %{customdata[10]:.1%}<br>"
                "Stored energy: %{customdata[11]:.2f} kWh"
                "<extra></extra>"
            ),
        )
    )
    target_trace_index = len(fig.data) - 1

    if show_cpa and math.isfinite(cpa_x_km) and math.isfinite(cpa_y_km):
        fig.add_trace(
            go.Scatter3d(
                x=[cpa_x_km],
                y=[cpa_y_km],
                z=[cpa_z_km],
                mode="markers+text",
                marker=dict(
                    size=10,
                    color="#FFFFFF",
                    symbol="x",
                ),
                text=["CPA"],
                textposition="bottom center",
                name="CPA",
                hovertemplate=(
                    f"CPA range: {geometry['cpa_range_m']:.1f} m"
                    "<extra></extra>"
                ),
            )
        )

    # 3-D 1σ transverse uncertainty ellipse in the plane normal to LOS.
    if show_uncertainty:
        sigma_az_m = float(
            row["Track Azimuth Cross-Range 1σ (m)"]
        )
        sigma_el_m = float(
            row["Track Elevation Cross-Range 1σ (m)"]
        )

        (
            _u_r,
            u_az,
            u_el,
            _range_m,
        ) = los_basis_from_position(
            p_sel
        )

        phi = np.linspace(
            0.0,
            2.0 * math.pi,
            121,
        )

        ellipse_points_m = np.array(
            [
                p_sel
                + sigma_az_m
                * math.cos(a)
                * u_az
                + sigma_el_m
                * math.sin(a)
                * u_el
                for a in phi
            ]
        )

        fig.add_trace(
            go.Scatter3d(
                x=ellipse_points_m[:, 0] / 1000.0,
                y=ellipse_points_m[:, 1] / 1000.0,
                z=ellipse_points_m[:, 2] / 1000.0,
                mode="lines",
                line=dict(
                    color="#FFB15A",
                    width=4,
                    dash="dot",
                ),
                name="Track 1σ",
                hoverinfo="skip",
            )
        )

    if show_beam_footprint:
        spot_diameter_m = float(
            row["Spot Diameter (m)"]
        )
        spot_radius_km = max(
            spot_diameter_m
            / 2.0
            / 1000.0,
            1e-6,
        )

        target_3d_km = np.array(
            [
                x_sel_km,
                y_sel_km,
                z_sel_km,
            ],
            dtype=float,
        )

        beam_axis = target_3d_km.copy()
        beam_axis_norm = max(
            float(
                np.linalg.norm(
                    beam_axis
                )
            ),
            1e-9,
        )
        b_hat = (
            beam_axis
            / beam_axis_norm
        )

        reference_axis = np.array(
            [0.0, 0.0, 1.0],
            dtype=float,
        )
        if abs(
            float(
                np.dot(
                    b_hat,
                    reference_axis,
                )
            )
        ) > 0.95:
            reference_axis = np.array(
                [0.0, 1.0, 0.0],
                dtype=float,
            )

        e1 = np.cross(
            b_hat,
            reference_axis,
        )
        e1 = e1 / max(
            float(
                np.linalg.norm(e1)
            ),
            1e-9,
        )
        e2 = np.cross(
            b_hat,
            e1,
        )
        e2 = e2 / max(
            float(
                np.linalg.norm(e2)
            ),
            1e-9,
        )

        beam_phi = np.linspace(
            0.0,
            2.0 * math.pi,
            121,
        )

        footprint_points = np.array(
            [
                target_3d_km
                + spot_radius_km
                * (
                    e1 * math.cos(a)
                    + e2 * math.sin(a)
                )
                for a in beam_phi
            ]
        )

        fig.add_trace(
            go.Scatter3d(
                x=footprint_points[:, 0],
                y=footprint_points[:, 1],
                z=footprint_points[:, 2],
                mode="lines",
                line=dict(
                    color=HUD_GREEN,
                    width=4,
                ),
                name="Effective Beam Footprint",
                hoverinfo="skip",
            )
        )

    if show_event_markers:
        event_specs = [
            (
                "DETECT",
                timeline["Detection Probability"].to_numpy(dtype=float) >= 0.50,
                HUD_GREEN_BRIGHT,
            ),
            (
                "TRACK VALID",
                timeline["Track Quality"].to_numpy(dtype=float) >= 0.60,
                "#FFF200",  # bright yellow event marker / label
            ),
            (
                "ENGAGE",
                timeline["Recommendation"].astype(str).str.contains(
                    "ENGAGE / CONTINUE",
                    regex=False,
                ).to_numpy(),
                HUD_RED,
            ),
            (
                "EFFECT INDEX 50%",
                timeline["Estimated Thermal Effect Index"].to_numpy(dtype=float) >= 0.50,
                HUD_ORANGE_BRIGHT,
            ),
        ]

        for label, mask, color in event_specs:
            matches = np.where(mask)[0]
            if len(matches) == 0:
                continue
            idx = int(matches[0])
            p_evt = positions_m[idx]
            fig.add_trace(
                go.Scatter3d(
                    x=[p_evt[0] / 1000.0],
                    y=[p_evt[1] / 1000.0],
                    z=[p_evt[2] / 1000.0],
                    mode="markers+text",
                    marker=dict(
                        size=9,
                        color=color,
                        symbol="diamond",
                    ),
                    text=[label],
                    textposition="bottom center",
                    name=label,
                    hovertemplate=(
                        f"{label}<br>"
                        f"t = {times_s[idx]:.2f} s"
                        "<extra></extra>"
                    ),
                )
            )

    # Animation updates target and beam from the authoritative timeline.
    if enable_animation and len(timeline) > 1:
        frames = []

        stride = max(
            1,
            int(
                math.ceil(
                    len(timeline) / 80
                )
            ),
        )
        animation_indices = list(
            range(
                0,
                len(timeline),
                stride,
            )
        )
        if animation_indices[-1] != len(timeline) - 1:
            animation_indices.append(
                len(timeline) - 1
            )

        for idx in animation_indices:
            p = positions_m[idx]
            px = float(p[0] / 1000.0)
            py = float(p[1] / 1000.0)
            pz = float(p[2] / 1000.0)
            r = timeline.iloc[idx]
            eff = float(
                r["Estimated Thermal Effect Index"]
            )
            if eff < 0.40:
                frame_color = HUD_GREEN_BRIGHT
            elif eff < 0.75:
                frame_color = HUD_ORANGE_BRIGHT
            else:
                frame_color = HUD_RED

            frames.append(
                go.Frame(
                    name=f"{idx}",
                    traces=[
                        beam_trace_index,
                        target_halo_trace_index,
                        target_trace_index,
                    ],
                    data=[
                        go.Scatter3d(
                            x=[0.0, px],
                            y=[0.0, py],
                            z=[0.0, pz],
                            mode="lines",
                            line=dict(
                                color=HUD_GREEN_BRIGHT,
                                width=max(
                                    4,
                                    min(
                                        12,
                                        4
                                        + 2
                                        * float(
                                            r["Spot Diameter (m)"]
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        go.Scatter3d(
                            x=[px],
                            y=[py],
                            z=[pz],
                            mode="markers",
                            marker=dict(
                                size=22,
                                color="rgba(0,0,0,0)",
                                symbol="circle",
                                line=dict(
                                    color=frame_color,
                                    width=5,
                                ),
                            ),
                            showlegend=False,
                        ),
                        go.Scatter3d(
                            x=[px],
                            y=[py],
                            z=[pz],
                            mode="markers+text",
                            marker=dict(
                                size=13,
                                color=CURRENT_TARGET_BLUE,
                                symbol="circle",
                                line=dict(
                                    color="#DCE7FF",
                                    width=1.5,
                                ),
                            ),
                            text=["Current Target"],
                            textposition="top center",
                            textfont=dict(
                                color=CURRENT_TARGET_BLUE,
                            ),
                            customdata=[[
                                float(r["Time (s)"]),
                                float(r["Range (km)"]),
                                float(r["LOS Rate (mrad/s)"]),
                                float(r["Altitude (m)"]),
                                float(r["Elevation Angle (deg)"]),
                                float(r["Track Angular 1σ (mrad)"]),
                                float(r["Atmospheric Transmission"]),
                                float(r["Average Irradiance (kW/m^2)"]),
                                float(r["Target ΔT (C)"]),
                                float(r["Estimated Thermal Effect Index"]),
                                float(r["Power Availability Ratio"]),
                                float(r["Stored Energy Remaining (kWh)"]),
                            ]],
                            hovertemplate=(
                                "<b>Current target</b><br>"
                                "Time: %{customdata[0]:.2f} s<br>"
                                "Range: %{customdata[1]:.2f} km<br>"
                                "LOS rate: %{customdata[2]:.3f} mrad/s<br>"
                                "Altitude: %{customdata[3]:.0f} m<br>"
                                "Elevation: %{customdata[4]:.2f}°<br>"
                                "Track 1σ: %{customdata[5]:.3f} mrad<br>"
                                "Atmospheric transmission: %{customdata[6]:.1%}<br>"
                                "Avg. irradiance: %{customdata[7]:.2f} kW/m²<br>"
                                "Target ΔT: %{customdata[8]:.1f} °C<br>"
                                "Thermal effect index: %{customdata[9]:.1%}<br>"
                                "Power availability: %{customdata[10]:.1%}<br>"
                                "Stored energy: %{customdata[11]:.2f} kWh"
                                "<extra></extra>"
                            ),
                        ),
                    ],
                )
            )

        fig.frames = frames

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.02,
                    y=1.08,
                    showactive=False,
                    bgcolor="rgba(2,4,2,0.80)",
                    bordercolor=HUD_ORANGE_DIM,
                    buttons=[
                        dict(
                            label="▶ PLAY",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {
                                        "duration": 100,
                                        "redraw": True,
                                    },
                                    "transition": {
                                        "duration": 0,
                                    },
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                },
                            ],
                        ),
                        dict(
                            label="■ PAUSE",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {
                                        "duration": 0,
                                        "redraw": False,
                                    },
                                    "transition": {
                                        "duration": 0,
                                    },
                                    "mode": "immediate",
                                },
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    x=0.15,
                    y=0.02,
                    len=0.70,
                    currentvalue=dict(
                        prefix="Frame: ",
                        font=dict(
                            color=HUD_ORANGE_BRIGHT,
                        ),
                    ),
                    steps=[
                        dict(
                            method="animate",
                            label=f"{timeline.iloc[idx]['Time (s)']:.1f}s",
                            args=[
                                [f"{idx}"],
                                {
                                    "mode": "immediate",
                                    "frame": {
                                        "duration": 0,
                                        "redraw": True,
                                    },
                                    "transition": {
                                        "duration": 0,
                                    },
                                },
                            ],
                        )
                        for idx in animation_indices
                    ],
                )
            ],
        )

    fig.update_layout(
        height=700,
        margin=dict(
            l=0,
            r=0,
            t=75,
            b=35,
        ),
        paper_bgcolor=HUD_BG,
        plot_bgcolor=HUD_BG,
        uirevision="directed-energy-3d-view",
        font=dict(
            color=HUD_TEXT,
            family="Consolas, monospace",
        ),
        title=dict(
            text=(
                "3D DIGITAL TWIN VIEW"
                " • "
                f"t = {t_sel:.2f} s"
            ),
            font=dict(
                color=HUD_ORANGE_BRIGHT,
                size=20,
            ),
        ),
        legend=dict(
            orientation="h",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(2,4,2,0.58)",
            bordercolor=HUD_GREEN_DIM,
            borderwidth=1,
            font=dict(size=10),
            itemsizing="constant",
        ),
        scene=dict(
            bgcolor=HUD_BG,
            aspectmode="manual",
            aspectratio=aspect_ratio,
            xaxis=dict(
                title="X / LOS Axis (km)",
                range=x_range,
                color=HUD_TEXT,
                gridcolor="#173817",
                zerolinecolor=HUD_GREEN_DIM,
                backgroundcolor=HUD_BG,
            ),
            yaxis=dict(
                title="Y / Cross-Range (km)",
                range=y_range,
                color=HUD_TEXT,
                gridcolor="#173817",
                zerolinecolor=HUD_GREEN_DIM,
                backgroundcolor=HUD_BG,
            ),
            zaxis=dict(
                title="Z / Altitude (km)",
                range=z_range,
                color=HUD_TEXT,
                gridcolor="#3A2814",
                zerolinecolor=HUD_ORANGE_DIM,
                backgroundcolor=HUD_BG,
            ),
            camera=camera,
        ),
    )

    return fig


# ============================================================
# UI
# ============================================================

st.markdown(
    f"""
    <div style="
        border:1px solid rgba(124,255,34,0.42);
        background:linear-gradient(180deg,rgba(6,17,6,.96),rgba(2,7,2,.96));
        border-radius:10px;
        padding:18px 22px;
        margin-bottom:14px;
        box-shadow:0 0 18px rgba(124,255,34,.06);
    ">
        <div style="
            font-size:2rem;
            font-weight:800;
            letter-spacing:.045em;
            color:{HUD_TEXT};
        ">
            DIRECTED ENERGY
        </div>
        <div style="
            font-size:1.65rem;
            font-weight:800;
            letter-spacing:.045em;
            color:{HUD_ORANGE_BRIGHT};
            margin-top:-2px;
        ">
            ENGAGEMENT DIGITAL TWIN
        </div>
        <div style="
            color:{HUD_MUTED};
            letter-spacing:.08em;
            margin-top:8px;
            font-size:.92rem;
        ">
            PHYSICS-INFORMED • CLOSED-LOOP • SYSTEMS ENGINEERING
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Low-order, non-classified digital engineering prototype. "
    "The model now includes Kalman covariance propagation, Beer-Lambert atmospheric "
    "extinction, and a lumped target thermal-response model. Outputs remain generic "
    "engineering estimates, not validated weapon-performance predictions."
)


with st.sidebar:
    st.header("Scenario Configuration")

    st.subheader("Environment")
    range_km = st.slider(
        "Initial horizontal range (km)",
        0.5,
        25.0,
        6.0,
        0.5,
    )
    humidity_pct = st.slider(
        "Relative humidity (%)",
        0,
        100,
        45,
    )
    visibility_km = st.slider(
        "Meteorological visibility (km)",
        1.0,
        40.0,
        20.0,
        1.0,
    )
    turbulence = st.slider(
        "Turbulence index",
        0.0,
        1.0,
        0.25,
        0.05,
    )
    wind_mps = st.slider(
        "Wind speed (m/s)",
        0.0,
        30.0,
        5.0,
        0.5,
    )
    ambient_temp_c = st.slider(
        "Ambient temperature (°C)",
        -20.0,
        60.0,
        25.0,
        1.0,
    )
    angstrom_exponent = st.slider(
        "Ångström aerosol exponent",
        0.0,
        2.5,
        1.3,
        0.1,
        help=(
            "Spectral aerosol-extinction scaling exponent. "
            "This remains a low-order atmospheric abstraction."
        ),
    )
    humidity_absorption_km_inv_at_100pct = st.slider(
        "Humidity absorption coefficient at 100% RH (1/km)",
        0.0,
        0.10,
        0.015,
        0.005,
        help=(
            "Generic absorption term. It is not a wavelength-resolved "
            "molecular spectroscopy model."
        ),
    )
    wind_pointing_sensitivity_urad_per_mps = st.slider(
        "Wind pointing sensitivity (µrad per m/s)",
        0.0,
        10.0,
        2.0,
        0.5,
    )

    st.subheader("Target")

    TARGET_PRESETS = {
        "Small Multirotor UAS": {
            "speed": 30.0, "angle": 30.0, "altitude": 150.0, "flight_path": -2.0,
            "aspect": 0.55, "maneuver": 0.75, "radius": 0.20,
            "absorptivity": 0.60, "areal_heat_capacity": 2.5,
            "thermal_loss": 0.010, "failure_delta_t": 140.0, "hardness": 0.8,
        },
        "Fixed-Wing UAS": {
            "speed": 55.0, "angle": 25.0, "altitude": 800.0, "flight_path": -3.0,
            "aspect": 0.68, "maneuver": 0.50, "radius": 0.30,
            "absorptivity": 0.55, "areal_heat_capacity": 3.5,
            "thermal_loss": 0.012, "failure_delta_t": 165.0, "hardness": 0.9,
        },
        "Large UAS": {
            "speed": 85.0, "angle": 15.0, "altitude": 2000.0, "flight_path": -2.0,
            "aspect": 0.80, "maneuver": 0.30, "radius": 0.60,
            "absorptivity": 0.50, "areal_heat_capacity": 5.0,
            "thermal_loss": 0.015, "failure_delta_t": 190.0, "hardness": 1.1,
        },
        "Loitering Munition": {
            "speed": 70.0, "angle": 15.0, "altitude": 1000.0, "flight_path": -8.0,
            "aspect": 0.72, "maneuver": 0.55, "radius": 0.22,
            "absorptivity": 0.52, "areal_heat_capacity": 3.8,
            "thermal_loss": 0.014, "failure_delta_t": 175.0, "hardness": 1.0,
        },
        "Cruise-Missile-Like Target": {
            "speed": 250.0, "angle": 8.0, "altitude": 300.0, "flight_path": -1.0,
            "aspect": 0.82, "maneuver": 0.28, "radius": 0.25,
            "absorptivity": 0.47, "areal_heat_capacity": 5.5,
            "thermal_loss": 0.018, "failure_delta_t": 215.0, "hardness": 1.25,
        },
        "Rocket / Artillery-Like Target": {
            "speed": 330.0, "angle": 5.0, "altitude": 3000.0, "flight_path": -25.0,
            "aspect": 0.85, "maneuver": 0.12, "radius": 0.18,
            "absorptivity": 0.44, "areal_heat_capacity": 6.2,
            "thermal_loss": 0.020, "failure_delta_t": 225.0, "hardness": 1.35,
        },
        "Helicopter-Like Target": {
            "speed": 70.0, "angle": 45.0, "altitude": 600.0, "flight_path": 0.0,
            "aspect": 0.88, "maneuver": 0.45, "radius": 0.85,
            "absorptivity": 0.48, "areal_heat_capacity": 7.0,
            "thermal_loss": 0.020, "failure_delta_t": 210.0, "hardness": 1.25,
        },
        "Fixed-Wing Aircraft": {
            "speed": 220.0, "angle": 40.0, "altitude": 3500.0, "flight_path": -3.0,
            "aspect": 0.90, "maneuver": 0.35, "radius": 1.10,
            "absorptivity": 0.46, "areal_heat_capacity": 8.0,
            "thermal_loss": 0.022, "failure_delta_t": 230.0, "hardness": 1.35,
        },
        "Generic Airborne Target": {
            "speed": 120.0, "angle": 45.0, "altitude": 1500.0, "flight_path": -5.0,
            "aspect": 0.75, "maneuver": 0.35, "radius": 0.40,
            "absorptivity": 0.50, "areal_heat_capacity": 4.0,
            "thermal_loss": 0.012, "failure_delta_t": 180.0, "hardness": 1.0,
        },
    }

    target_type = st.selectbox(
        "Target type",
        list(TARGET_PRESETS.keys()),
    )
    st.caption(
        "Target presets are generic demonstration parameters, not authoritative threat specifications."
    )
    preset = TARGET_PRESETS[target_type]

    if st.session_state.get(
        "_last_target_type_v4"
    ) != target_type:
        st.session_state.target_speed_v4 = preset["speed"]
        st.session_state.target_angle_v4 = preset["angle"]
        st.session_state.target_altitude_v4 = preset["altitude"]
        st.session_state.target_flight_path_v4 = preset["flight_path"]
        st.session_state.target_aspect_v4 = preset["aspect"]
        st.session_state.target_maneuver_v4 = preset["maneuver"]
        st.session_state.target_radius_v4 = preset["radius"]
        st.session_state.target_absorptivity_v4 = preset["absorptivity"]
        st.session_state.target_areal_heat_capacity_v4 = preset["areal_heat_capacity"]
        st.session_state.target_thermal_loss_v4 = preset["thermal_loss"]
        st.session_state.target_failure_delta_t_v4 = preset["failure_delta_t"]
        st.session_state.target_hardness_v4 = preset["hardness"]
        st.session_state._last_target_type_v4 = target_type

    speed_mps = st.slider(
        "Target speed (m/s)",
        10.0,
        350.0,
        key="target_speed_v4",
        step=5.0,
    )
    velocity_angle_deg = st.slider(
        "Horizontal velocity angle relative to LOS (deg)",
        0.0,
        180.0,
        key="target_angle_v4",
        step=5.0,
        help="0° = horizontally closing, 90° = crossing, 180° = receding.",
    )
    initial_altitude_m = st.slider(
        "Initial target altitude (m)",
        0.0,
        10000.0,
        key="target_altitude_v4",
        step=100.0,
        help="Physical z-coordinate used by the authoritative 3-D engagement model.",
    )
    flight_path_angle_deg = st.slider(
        "Flight-path angle (deg)",
        -60.0,
        45.0,
        key="target_flight_path_v4",
        step=1.0,
        help=(
            "Negative = descending, 0° = level, positive = climbing. "
            "Descending trajectories terminate at the z=0 reference plane."
        ),
    )
    aspect_factor = st.slider(
        "Aspect / observability factor",
        0.2,
        1.0,
        key="target_aspect_v4",
        step=0.05,
    )
    maneuver_factor = st.slider(
        "Maneuver / acceleration uncertainty index",
        0.0,
        1.0,
        key="target_maneuver_v4",
        step=0.05,
    )
    characteristic_radius_m = st.slider(
        "Characteristic aimpoint radius (m)",
        0.05,
        2.0,
        key="target_radius_v4",
        step=0.05,
        help="Synthetic characteristic target radius used only for pointing-tolerance normalization.",
    )

    with st.expander("Target thermal properties"):
        absorptivity = st.slider(
            "Absorptivity",
            0.05,
            0.95,
            key="target_absorptivity_v4",
            step=0.05,
        )
        areal_heat_capacity_kj_m2k = st.slider(
            "Areal heat capacity (kJ/m²-K)",
            0.5,
            20.0,
            key="target_areal_heat_capacity_v4",
            step=0.5,
        )
        thermal_loss_coeff_kw_m2k = st.slider(
            "Effective thermal-loss coefficient (kW/m²-K)",
            0.0,
            0.10,
            key="target_thermal_loss_v4",
            step=0.005,
        )
        failure_delta_t_c = st.slider(
            "Reference failure ΔT (°C)",
            50.0,
            600.0,
            key="target_failure_delta_t_v4",
            step=10.0,
            help="Synthetic reference used to normalize the thermal-effect index.",
        )
        hardness_multiplier = st.slider(
            "Hardness multiplier",
            0.5,
            3.0,
            key="target_hardness_v4",
            step=0.1,
        )

    st.subheader("Sensors / State Estimation")
    radar_quality = st.slider(
        "Radar quality",
        0.0,
        1.0,
        0.90,
        0.05,
    )
    eo_ir_quality = st.slider(
        "EO/IR quality",
        0.0,
        1.0,
        0.85,
        0.05,
    )
    data_latency_ms = st.slider(
        "Data latency (ms)",
        0,
        1500,
        120,
        10,
    )
    dropped_measurement_rate = st.slider(
        "Dropped measurement rate",
        0.0,
        0.5,
        0.03,
        0.01,
    )
    track_update_hz = st.slider(
        "Track update rate (Hz)",
        1.0,
        50.0,
        10.0,
        1.0,
    )
    range_measurement_sigma_m = st.slider(
        "Range measurement 1σ (m)",
        0.5,
        100.0,
        10.0,
        0.5,
    )
    bearing_measurement_sigma_mrad = st.slider(
        "Bearing measurement 1σ (mrad)",
        0.01,
        2.0,
        0.20,
        0.01,
    )
    process_accel_sigma_mps2 = st.slider(
        "Process acceleration 1σ (m/s²)",
        0.1,
        30.0,
        3.0,
        0.1,
        help="Generic process-noise term for the constant-velocity covariance model.",
    )

    st.subheader("HEL Subsystem")
    requested_optical_source_power_kw = st.slider(
        "Requested optical source power (kW)",
        10.0,
        500.0,
        100.0,
        10.0,
    )
    wall_plug_efficiency = st.slider(
        "Wall-plug efficiency",
        0.10,
        0.70,
        0.35,
        0.05,
    )
    optics_efficiency = st.slider(
        "Optical-train efficiency",
        0.20,
        1.00,
        0.80,
        0.05,
    )
    commanded_dwell_time_s = st.slider(
        "Commanded dwell time (s)",
        0.1,
        20.0,
        4.0,
        0.1,
    )
    wavelength_um = st.slider(
        "Optical wavelength (µm)",
        0.5,
        2.0,
        1.06,
        0.01,
    )
    beam_quality_m2 = st.slider(
        "Beam quality M²",
        1.0,
        5.0,
        1.3,
        0.1,
        help="M² = 1 is ideal Gaussian-beam quality.",
    )
    additional_half_angle_divergence_mrad = st.slider(
        "Additional half-angle divergence (mrad)",
        0.0,
        1.0,
        0.05,
        0.01,
    )
    initial_beam_diameter_m = st.slider(
        "Initial beam diameter (m)",
        0.05,
        1.00,
        0.20,
        0.05,
    )
    base_pointing_jitter_mrad = st.slider(
        "Base pointing jitter 1σ (mrad)",
        0.00,
        0.50,
        0.03,
        0.01,
    )
    beam_director_max_rate_mrad_s = st.slider(
        "Beam-director max LOS rate (mrad/s)",
        10.0,
        500.0,
        150.0,
        10.0,
        help=(
            "Generic beam-director angular-rate limit. Scenarios that exceed this "
            "rate incur servo pointing error and can trigger a HOLD recommendation."
        ),
    )
    beam_director_servo_time_constant_s = st.slider(
        "Beam-director servo time constant (ms)",
        0.5,
        20.0,
        3.0,
        0.5,
        help=(
            "Generic first-order servo lag used to convert LOS angular rate into "
            "additional pointing error."
        ),
    ) / 1000.0

    st.subheader("Platform")
    stored_energy_kwh = st.slider(
        "Stored energy (kWh)",
        1.0,
        100.0,
        20.0,
        1.0,
    )
    storage_max_discharge_kw = st.slider(
        "Storage max discharge power (kW)",
        0.0,
        5000.0,
        1000.0,
        50.0,
    )
    generator_power_kw = st.slider(
        "Generator power (kW)",
        0.0,
        500.0,
        150.0,
        10.0,
    )
    cooling_capacity_kw = st.slider(
        "Cooling capacity (kW)",
        0.0,
        300.0,
        70.0,
        5.0,
    )
    coolant_temp_c = st.slider(
        "Initial coolant temperature (°C)",
        10.0,
        80.0,
        30.0,
        1.0,
    )
    thermal_limit_c = st.slider(
        "Thermal limit (°C)",
        40.0,
        120.0,
        80.0,
        1.0,
    )
    thermal_capacitance_kj_per_c = st.slider(
        "Thermal capacitance (kJ/°C)",
        50.0,
        2000.0,
        500.0,
        50.0,
    )
    subsystem_health = st.slider(
        "Subsystem health",
        0.5,
        1.0,
        0.97,
        0.01,
    )


    st.subheader("3D Digital Twin View")
    st.caption(
        "Altitude is now part of the authoritative 3-D target state and directly "
        "affects slant range, LOS geometry, tracking, beam pointing, and propagation."
    )


env = Environment(
    range_km,
    humidity_pct,
    visibility_km,
    turbulence,
    wind_mps,
    ambient_temp_c,
    angstrom_exponent,
    humidity_absorption_km_inv_at_100pct,
    wind_pointing_sensitivity_urad_per_mps,
)

tgt = Target(
    target_type,
    speed_mps,
    velocity_angle_deg,
    initial_altitude_m,
    flight_path_angle_deg,
    aspect_factor,
    maneuver_factor,
    characteristic_radius_m,
    absorptivity,
    areal_heat_capacity_kj_m2k,
    thermal_loss_coeff_kw_m2k,
    failure_delta_t_c,
    hardness_multiplier,
)

sensors = SensorState(
    radar_quality,
    eo_ir_quality,
    data_latency_ms,
    dropped_measurement_rate,
    track_update_hz,
    range_measurement_sigma_m,
    bearing_measurement_sigma_mrad,
    process_accel_sigma_mps2,
)

hel = HELState(
    requested_optical_source_power_kw,
    wall_plug_efficiency,
    optics_efficiency,
    commanded_dwell_time_s,
    wavelength_um,
    beam_quality_m2,
    additional_half_angle_divergence_mrad,
    initial_beam_diameter_m,
    base_pointing_jitter_mrad,
    beam_director_max_rate_mrad_s,
    beam_director_servo_time_constant_s,
)

platform = PlatformState(
    stored_energy_kwh,
    storage_max_discharge_kw,
    generator_power_kw,
    cooling_capacity_kw,
    coolant_temp_c,
    thermal_limit_c,
    thermal_capacitance_kj_per_c,
    subsystem_health,
)

timeline, result = simulate_time_stepped_engagement(
    env,
    tgt,
    sensors,
    hel,
    platform,
    dt_s=0.10,
)


# ============================================================
# Engagement state
# ============================================================

st.subheader("Engagement State")

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Track Quality",
    f"{result['Track Quality'] * 100:.1f}%",
)
m2.metric(
    "Track Angular 1σ",
    f"{result['Track Angular 1σ (mrad)']:.3f} mrad",
)
m3.metric(
    "Atmospheric Transmission",
    f"{result['Atmospheric Transmission'] * 100:.1f}%",
)
m4.metric(
    "Aimpoint Margin Index",
    f"{result['Aimpoint Margin Index'] * 100:.1f}%",
)

m5, m6, m7, m8 = st.columns(4)
m5.metric(
    "Average Target Irradiance",
    f"{result['Average Irradiance (kW/m^2)']:.1f} kW/m²",
)
m6.metric(
    "Target Surface ΔT",
    f"{result['Target ΔT (C)']:.1f} °C",
)
m7.metric(
    "Thermal Effect Index",
    f"{result['Estimated Thermal Effect Index'] * 100:.1f}%",
)
m8.metric(
    "Readiness Score",
    f"{result['Readiness Score'] * 100:.1f}%",
)

if "ENGAGE" in result["Recommendation"]:
    st.success(result["Recommendation"])
elif "CAUTION" in result["Recommendation"]:
    st.warning(result["Recommendation"])
else:
    st.error(result["Recommendation"])


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Engagement Loop",
        "State Estimation",
        "Time History",
        "Monte Carlo",
        "Model State",
        "3D Digital Twin",
        "Export",
    ]
)


# ============================================================
# Engagement loop
# ============================================================

with tab1:
    st.markdown("### End-to-End Engagement Architecture")

    stages = [
        (
            "1. Detect",
            result["Detection Probability"],
        ),
        (
            "2. Identify",
            result["Classification Confidence"],
        ),
        (
            "3. Track",
            result["Track Quality"],
        ),
        (
            "4. Decide",
            result["Readiness Score"],
        ),
        (
            "5. Point / Aim",
            result["Aimpoint Margin Index"],
        ),
        (
            "6. Deliver Energy",
            result["Power Availability Ratio"]
            * result["Atmospheric Transmission"],
        ),
        (
            "7. Assess Effect",
            result["Estimated Thermal Effect Index"],
        ),
        (
            "8. Re-engage",
            result["Readiness Score"],
        ),
    ]

    stage_df = pd.DataFrame(
        stages,
        columns=[
            "Stage",
            "State / Confidence",
        ],
    )

    st.dataframe(
        stage_df.style.format({
            "State / Confidence": "{:.1%}"
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Decision-Support State")

    state_table = pd.DataFrame({
        "Parameter": [
            "Detection probability",
            "Classification confidence",
            "Track quality",
            "Track cross-range 1σ",
            "Track angular 1σ",
            "LOS rate",
            "Aimpoint margin index",
            "Atmospheric transmission",
            "Optical depth",
            "Available engagement time",
            "Time to CPA",
            "Time to ground impact",
            "CPA range",
            "Effective dwell time",
            "Requested electrical input",
            "Actual electrical input",
            "Power availability ratio",
            "Generator contribution",
            "Storage draw",
            "Target optical power",
            "Spot diameter",
            "Average target irradiance",
            "Absorbed heat flux",
            "Target surface ΔT",
            "Target surface temperature",
            "Estimated thermal effect index",
            "Coolant temperature after dwell",
            "Thermal margin",
            "Energy margin",
            "Recommendation",
        ],
        "Value": [
            f"{result['Detection Probability']:.1%}",
            f"{result['Classification Confidence']:.1%}",
            f"{result['Track Quality']:.1%}",
            f"{result['Track Cross-Range 1σ (m)']:.2f} m",
            f"{result['Track Angular 1σ (mrad)']:.3f} mrad",
            f"{result['LOS Rate (mrad/s)']:.3f} mrad/s",
            f"{result['Aimpoint Margin Index']:.1%}",
            f"{result['Atmospheric Transmission']:.1%}",
            f"{result['Optical Depth']:.3f}",
            f"{result['Available Engagement Time (s)']:.2f} s",
            f"{result['Time to CPA (s)']:.2f} s",
            (
                f"{result['Time to Ground Impact (s)']:.2f} s"
                if math.isfinite(result["Time to Ground Impact (s)"])
                else "N/A"
            ),
            f"{result['CPA Range (m)']:.1f} m",
            f"{result['Effective Dwell Time (s)']:.2f} s",
            f"{result['Requested Electrical Input (kW)']:.1f} kW",
            f"{result['Actual Electrical Input (kW)']:.1f} kW",
            f"{result['Power Availability Ratio']:.1%}",
            f"{result['Generator Contribution (kW)']:.1f} kW",
            f"{result['Storage Draw (kW)']:.1f} kW",
            f"{result['Target Optical Power (kW)']:.1f} kW",
            f"{result['Spot Diameter (m)']:.2f} m",
            f"{result['Average Irradiance (kW/m^2)']:.1f} kW/m²",
            f"{result['Absorbed Heat Flux (kW/m^2)']:.1f} kW/m²",
            f"{result['Target ΔT (C)']:.1f} °C",
            f"{result['Target Surface Temp (C)']:.1f} °C",
            f"{result['Estimated Thermal Effect Index']:.1%}",
            f"{result['Coolant Temp After Dwell (C)']:.1f} °C",
            f"{result['Thermal Margin']:.1%}",
            f"{result['Energy Margin']:.1%}",
            result["Recommendation"],
        ],
    })

    st.dataframe(
        state_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# State estimation
# ============================================================

with tab2:
    st.markdown("### Covariance-Based State Estimation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Radial Position 1σ",
        f"{result['Track Radial 1σ (m)']:.2f} m",
    )
    c2.metric(
        "Cross-Range 1σ",
        f"{result['Track Cross-Range 1σ (m)']:.2f} m",
    )
    c3.metric(
        "Angular Track 1σ",
        f"{result['Track Angular 1σ (mrad)']:.3f} mrad",
    )
    c4.metric(
        "Target Angular Radius",
        f"{result['Target Angular Radius (mrad)']:.3f} mrad",
    )

    st.caption(
        "The tracker is a sequential constant-velocity Kalman covariance model. "
        "Covariance is carried continuously through the engagement and propagated "
        "through processing latency. It does not represent a particular operational "
        "radar, EO/IR tracker, or fire-control system."
    )

    uncertainty_df = pd.DataFrame({
        "Quantity": [
            "Range measurement 1σ",
            "Bearing measurement 1σ",
            "Process acceleration 1σ",
            "Track update rate",
            "Data latency",
            "LOS angular rate",
            "Azimuth LOS rate",
            "Elevation LOS rate",
            "Elevation angle",
            "Measurement availability",
            "Beam-director rate utilization",
            "Servo tracking error",
            "Effective pointing error",
        ],
        "Value": [
            f"{sensors.range_measurement_sigma_m:.2f} m",
            f"{sensors.bearing_measurement_sigma_mrad:.3f} mrad",
            f"{sensors.process_accel_sigma_mps2:.2f} m/s²",
            f"{sensors.track_update_hz:.1f} Hz",
            f"{sensors.data_latency_ms:.0f} ms",
            f"{result['LOS Rate (mrad/s)']:.3f} mrad/s",
            f"{result['Azimuth LOS Rate (mrad/s)']:.3f} mrad/s",
            f"{result['Elevation LOS Rate (mrad/s)']:.3f} mrad/s",
            f"{result['Elevation Angle (deg)']:.2f}°",
            f"{result['Measurement Availability']:.1%}",
            f"{result['Beam Director Rate Utilization']:.1%}",
            f"{result['Servo Tracking Error (mrad)']:.3f} mrad",
            f"{result['Effective Pointing Error (mrad)']:.3f} mrad",
        ],
    })

    st.dataframe(
        uncertainty_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Time histories
# ============================================================

with tab3:
    # Reuse the authoritative engagement timeline generated for the dashboard.
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(
        timeline["Time (s)"],
        timeline["Estimated Thermal Effect Index"],
        color=HUD_GREEN,
        linewidth=2.2,
    )
    ax1.set_xlabel("Dwell Time (s)")
    ax1.set_ylabel("Estimated Thermal Effect Index")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)


    fig_range, ax_range = plt.subplots(figsize=(9, 4))
    ax_range.plot(
        timeline["Time (s)"],
        timeline["Range (km)"],
        color=HUD_ORANGE,
        linewidth=2.2,
    )
    ax_range.set_xlabel("Engagement Time (s)")
    ax_range.set_ylabel("Target Range (km)")
    ax_range.grid(True, alpha=0.3)
    st.pyplot(fig_range)
    plt.close(fig_range)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(
        timeline["Time (s)"],
        timeline["Target Surface Temp (C)"],
        color=HUD_ORANGE,
        linewidth=2.2,
        label="Target surface",
    )
    ax2.plot(
        timeline["Time (s)"],
        timeline["Coolant Temp (C)"],
        color=HUD_GREEN,
        linewidth=1.8,
        label="Platform coolant",
    )
    ax2.axhline(
        env.ambient_temp_c,
        linestyle=":",
        color=HUD_MUTED,
        linewidth=1.2,
        label="Ambient",
    )
    ax2.set_xlabel("Dwell Time (s)")
    ax2.set_ylabel("Temperature (°C)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.plot(
        timeline["Time (s)"],
        timeline["Stored Energy Remaining (kWh)"],
        color=HUD_GREEN,
        linewidth=2.2,
    )
    ax3.set_xlabel("Dwell Time (s)")
    ax3.set_ylabel("Stored Energy Remaining (kWh)")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    plt.close(fig3)


# ============================================================
# Monte Carlo
# ============================================================

with tab4:
    st.markdown("### Monte Carlo Uncertainty Analysis")

    runs = st.slider(
        "Simulation runs",
        100,
        3000,
        1000,
        100,
    )

    if st.button("Run Monte Carlo"):
        records = [
            simulate_dynamic_monte_carlo_run(
                env,
                tgt,
                sensors,
                hel,
                platform,
            )
            for _ in range(runs)
        ]

        mc = pd.DataFrame(records)
        st.session_state["directed_energy_mc_results"] = mc.copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Mean Thermal Effect Index",
            f"{mc['Estimated Thermal Effect Index'].mean():.1%}",
        )
        c2.metric(
            "P(Index > 70%)",
            f"{(mc['Estimated Thermal Effect Index'] > 0.70).mean():.1%}",
        )
        c3.metric(
            "Mean Track Angular 1σ",
            f"{mc['Track Angular 1σ (mrad)'].mean():.3f} mrad",
        )
        c4.metric(
            "Thermal Constraint Rate",
            f"{(mc['Thermal Margin'] < 0.12).mean():.1%}",
        )

        fig4, ax4 = plt.subplots(figsize=(9, 4))
        ax4.hist(
            mc["Estimated Thermal Effect Index"],
            bins=30,
            color=HUD_GREEN_DIM,
            edgecolor=HUD_GREEN_BRIGHT,
        )
        ax4.set_xlabel("Estimated Thermal Effect Index")
        ax4.set_ylabel("Frequency")
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        plt.close(fig4)

        summary = mc[[
            "Detection Probability",
            "Track Quality",
            "Track Angular 1σ (mrad)",
            "Aimpoint Margin Index",
            "Atmospheric Transmission",
            "Effective Dwell Time (s)",
            "Target Optical Power (kW)",
            "Spot Diameter (m)",
            "Average Irradiance (kW/m^2)",
            "Target ΔT (C)",
            "Estimated Thermal Effect Index",
            "Thermal Margin",
            "Energy Margin",
            "Readiness Score",
        ]].describe().T

        st.dataframe(
            summary,
            use_container_width=True,
        )

        csv = mc.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Monte Carlo CSV",
            csv,
            file_name="directed_energy_monte_carlo.csv",
            mime="text/csv",
        )


# ============================================================
# Model state
# ============================================================

with tab5:
    st.markdown("### Digital Twin State Vector")

    st.json({
        "Environment": asdict(env),
        "Target": asdict(tgt),
        "Sensor State": asdict(sensors),
        "HEL State": asdict(hel),
        "Platform State": asdict(platform),
        "Engagement Output": result,
    })


# ============================================================
# 3D digital twin visualization
# ============================================================

with tab6:
    st.markdown("### Interactive 3D Digital Twin")

    st.caption(
        "This view renders the authoritative true 3-D constant-velocity engagement "
        "solution. The target z-coordinate is physically modeled and participates in "
        "slant range, CPA, LOS azimuth/elevation, tracking covariance, beam pointing, "
        "atmospheric path length, irradiance, and thermal calculations."
    )

    if timeline is not None and not timeline.empty:
        ctrl1, ctrl2, ctrl3 = st.columns([1.1, 1.0, 1.2])

        with ctrl1:
            selected_step = st.slider(
                "3D engagement time step",
                0,
                len(timeline) - 1,
                len(timeline) - 1,
                1,
            )

            camera_preset = st.selectbox(
                "Camera preset",
                [
                    "Isometric",
                    "Top-Down",
                    "Tactical",
                    "Beam Sight",
                    "Target Chase",
                ],
            )

        with ctrl2:
            enable_animation = st.toggle(
                "Playback animation",
                value=True,
            )
            show_target_path = st.toggle(
                "Target path",
                value=True,
            )
            show_cpa = st.toggle(
                "CPA marker",
                value=True,
            )
            show_engagement_zone = st.toggle(
                "Engagement zone",
                value=True,
            )

        with ctrl3:
            show_uncertainty = st.toggle(
                "Track 1σ overlay",
                value=True,
            )
            show_beam_footprint = st.toggle(
                "Beam footprint",
                value=True,
            )
            show_event_markers = st.toggle(
                "Engagement events",
                value=True,
            )

        row_3d = timeline.iloc[selected_step]

        d1, d2, d3, d4, d5, d6 = st.columns(6)

        d1.metric(
            "Time",
            f"{row_3d['Time (s)']:.2f} s",
        )
        d2.metric(
            "Slant Range",
            f"{row_3d['Range (km)']:.2f} km",
        )
        d3.metric(
            "Altitude",
            f"{row_3d['Altitude (m)']:.0f} m",
        )
        d4.metric(
            "Elevation",
            f"{row_3d['Elevation Angle (deg)']:.2f}°",
        )
        d5.metric(
            "LOS Rate",
            f"{row_3d['LOS Rate (mrad/s)']:.3f} mrad/s",
        )
        d6.metric(
            "Thermal Effect",
            f"{row_3d['Estimated Thermal Effect Index']:.1%}",
        )

        twin_3d = build_3d_digital_twin_figure(
            timeline,
            env,
            tgt,
            result,
            selected_step,
            camera_preset=camera_preset,
            show_target_path=show_target_path,
            show_cpa=show_cpa,
            show_uncertainty=show_uncertainty,
            show_beam_footprint=show_beam_footprint,
            show_engagement_zone=show_engagement_zone,
            show_event_markers=show_event_markers,
            enable_animation=enable_animation,
        )

        st.plotly_chart(
            twin_3d,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True,
            },
        )

        t1, t2, t3, t4 = st.columns(4)
        t1.metric(
            "Atmospheric Transmission",
            f"{row_3d['Atmospheric Transmission']:.1%}",
        )
        t2.metric(
            "Average Irradiance",
            f"{row_3d['Average Irradiance (kW/m^2)']:.2f} kW/m²",
        )
        t3.metric(
            "Target ΔT",
            f"{row_3d['Target ΔT (C)']:.1f} °C",
        )
        t4.metric(
            "Stored Energy",
            f"{row_3d['Stored Energy Remaining (kWh)']:.2f} kWh",
        )

        st.caption(
            "3D symbology: blue = Current Target identity; the surrounding target-state "
            "halo changes green → fire orange → red as the Estimated Thermal Effect Index "
            "increases. DETECT is green, TRACK VALID is bright yellow, ENGAGE is bright red, "
            "orange = target path, bright green = beam/LOS, orange dotted ellipse = approximate "
            "1σ track uncertainty, green ring = effective beam footprint, and white X = CPA."
        )
    else:
        st.info(
            "No finite engagement timeline is available for the current scenario."
        )



# ============================================================
# Export
# ============================================================

with tab7:
    st.markdown("### Export Simulation Data")
    st.caption(
        "Exports are generated from the same authoritative 3-D engagement state used "
        "by the dashboard and digital twin. Export operations do not modify the model."
    )

    preferred_columns = [
        "Time (s)",
        "Range (km)",
        "Physics Evaluation Range (km)",
        "X (km)",
        "Y (km)",
        "Z (km)",
        "Altitude (m)",
        "Azimuth (deg)",
        "Elevation Angle (deg)",
        "Detection Probability",
        "Classification Confidence",
        "Measurement Availability",
        "Track Quality",
        "Track Radial 1σ (m)",
        "Track Azimuth Cross-Range 1σ (m)",
        "Track Elevation Cross-Range 1σ (m)",
        "Track Angular 1σ (mrad)",
        "Track Azimuth 1σ (mrad)",
        "Track Elevation 1σ (mrad)",
        "LOS Rate (mrad/s)",
        "Azimuth LOS Rate (mrad/s)",
        "Elevation LOS Rate (mrad/s)",
        "Aimpoint Margin Index",
        "Atmospheric Transmission",
        "Aerosol Extinction (1/km)",
        "Rayleigh Extinction (1/km)",
        "Humidity Extinction (1/km)",
        "Optical Depth",
        "Available Engagement Time (s)",
        "Time to CPA (s)",
        "Time to Ground Impact (s)",
        "CPA Range (m)",
        "Effective Dwell Time (s)",
        "Requested Optical Source Power (kW)",
        "Actual Optical Source Power (kW)",
        "Requested Electrical Input (kW)",
        "Actual Electrical Input (kW)",
        "Power Availability Ratio",
        "Generator Contribution (kW)",
        "Storage Draw (kW)",
        "Target Optical Power (kW)",
        "Diffraction Half-Angle (mrad)",
        "Effective Beam Half-Angle (mrad)",
        "Effective Pointing Error (mrad)",
        "Stochastic Pointing 1σ (mrad)",
        "Servo Tracking Error (mrad)",
        "Azimuth Servo Error (mrad)",
        "Elevation Servo Error (mrad)",
        "Beam Director Rate Utilization",
        "Beam Director Rate Demand Ratio",
        "Beam Director Rate Saturated",
        "Spot Diameter (m)",
        "Average Irradiance (kW/m^2)",
        "Absorbed Heat Flux (kW/m^2)",
        "Absorbed Exposure (kJ/m^2)",
        "Target ΔT (C)",
        "Target Surface Temp (C)",
        "Estimated Thermal Effect Index",
        "Coolant Temp (C)",
        "Thermal Margin",
        "Energy Margin",
        "Storage Energy Used (kWh)",
        "Stored Energy Remaining (kWh)",
        "Readiness Score",
        "Recommendation",
    ]

    export_columns = [
        col for col in preferred_columns
        if col in timeline.columns
    ]
    export_timeline = timeline[export_columns].copy()

    timeline_csv = export_timeline.to_csv(index=False).encode("utf-8")

    export_package = {
        "schema_version": "1.0",
        "model": "Directed Energy Engagement Digital Twin",
        "model_scope": (
            "Low-order, non-classified digital engineering prototype. "
            "Outputs are generic engineering estimates and are not validated "
            "weapon-performance predictions."
        ),
        "environment": asdict(env),
        "target": asdict(tgt),
        "sensor_state": asdict(sensors),
        "hel_state": asdict(hel),
        "platform_state": asdict(platform),
        "final_engagement_state": result,
    }

    scenario_json = json.dumps(
        export_package,
        indent=2,
        default=str,
    ).encode("utf-8")

    e1, e2 = st.columns(2)

    with e1:
        st.download_button(
            "Download Time-History CSV",
            timeline_csv,
            file_name="directed_energy_time_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with e2:
        st.download_button(
            "Download Scenario + Final State JSON",
            scenario_json,
            file_name="directed_energy_scenario_final_state.json",
            mime="application/json",
            use_container_width=True,
        )

    mc_export = st.session_state.get("directed_energy_mc_results")

    if isinstance(mc_export, pd.DataFrame) and not mc_export.empty:
        mc_csv = mc_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Latest Monte Carlo CSV",
            mc_csv,
            file_name="directed_energy_monte_carlo.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            f"Latest Monte Carlo export contains {len(mc_export):,} simulation runs."
        )
    else:
        st.info(
            "Run the Monte Carlo analysis to enable the Monte Carlo CSV export."
        )

    st.markdown("#### Time-History Export Preview")
    st.dataframe(
        export_timeline.head(25),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Export Metadata")
    export_metadata = pd.DataFrame(
        {
            "Item": [
                "Target type",
                "Initial horizontal range",
                "Initial altitude",
                "Flight-path angle",
                "Commanded dwell",
                "Timeline rows",
                "Final recommendation",
            ],
            "Value": [
                tgt.target_type,
                f"{env.range_km:.2f} km",
                f"{tgt.initial_altitude_m:.0f} m",
                f"{tgt.flight_path_angle_deg:.1f}°",
                f"{hel.commanded_dwell_time_s:.2f} s",
                f"{len(export_timeline):,}",
                result["Recommendation"],
            ],
        }
    )
    st.dataframe(
        export_metadata,
        use_container_width=True,
        hide_index=True,
    )


st.divider()

st.caption(
    "Engineering note: This application is a physics-informed, low-order digital "
    "engineering framework. Atmospheric extinction uses a Beer-Lambert model with "
    "visibility-derived aerosol extinction, Rayleigh scaling, and a generic humidity "
    "term. Tracking uses sequential constant-velocity Kalman covariance propagation, "
    "and azimuth/elevation LOS rates are computed directly from the evolving 3-D geometry. "
    "Target response uses a lumped areal thermal model. The 3-D view renders the same "
    "modeled x/y/z constant-velocity target state used by the engagement physics, including "
    "slant range, CPA, LOS geometry, covariance projection, and beam pointing. The model "
    "does not include full aerodynamic flight dynamics or target-specific guidance laws. "
    "None of these models constitute validated "
    "operational weapon-performance, lethality, or probability-of-kill predictions. "
    "The current target preset library spans multiple generic airborne target classes, "
    "and the target-speed envelope remains intentionally limited to 350 m/s."
)
