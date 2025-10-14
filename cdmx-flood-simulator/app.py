#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CDMX — Rain fill & Desazolve simulator (Streamlit)
# - Bundles a default dataset at data/atlas-de-riesgo-inundaciones.csv
# - Outputs boundary map, baseline / desazolve / impact maps, and GIF+MP4 animation.
#
# Install:  pip install -r requirements.txt
# Run:      streamlit ejecutar app.py

import io
import json
import os
import warnings
import logging
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Try to use a bundled ffmpeg if available (works on Streamlit Cloud and local)
try:
    import imageio_ffmpeg  # pip install imageio-ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

# ---------------------- App globals & noise control ----------------------

APP_DIR = pathlib.Path(__file__).parent
DEFAULT_ATLAS_PATH = APP_DIR / "data" / "atlas-de-riesgo-inundaciones.csv"

# Silence common warnings/log spam
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# Must be the first Streamlit call
st.set_page_config(page_title="CDMX — Simulador de Lluvia y Desazolve (barquito)", layout="wide")


# ---------------------- Geometry & data structures ----------------------

@dataclass
class Zone:
    alcaldia: str
    int2: int
    area_m2: float
    coords: np.ndarray    # (N,2) lon,lat; closed
    centroid: Tuple[float, float]


def _ensure_closed(arr: np.ndarray) -> np.ndarray:
    if not np.allclose(arr[0], arr[-1]):
        arr = np.vstack([arr, arr[0]])
    return arr


def _ring_centroid(coords: np.ndarray) -> Tuple[float, float]:
    x = coords[:, 0]; y = coords[:, 1]
    a = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if abs(a) < 1e-12:
        return float(x.mean()), float(y.mean())
    cx = (1.0 / (6.0 * a)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    cy = (1.0 / (6.0 * a)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    return float(cx), float(cy)


def _make_patch(coords: np.ndarray, **kw) -> PathPatch:
    path = Path(
        coords,
        np.concatenate([[Path.MOVETO], np.full(len(coords) - 1, Path.LINETO, dtype=Path.code_type)])
    )
    return PathPatch(path, **kw)


# ---------------------- Loaders ----------------------

@st.cache_data(show_spinner=False)
def load_zones_from_csv_bytes(csv_bytes: bytes) -> List[Zone]:
    """Parse polygons from a CSV that has a 'geo_shape' Polygon column and fields:
       alcaldia, int2, area_m2. Returns a list of Zone objects."""
    df = pd.read_csv(io.BytesIO(csv_bytes))
    req = ["alcaldia", "int2", "area_m2", "geo_shape"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"CSV is missing required column: {c}")

    zonas: List[Zone] = []
    for _, r in df.iterrows():
        try:
            g = json.loads(r["geo_shape"])
            if g.get("type") != "Polygon":
                continue
            coords = np.asarray(g["coordinates"][0], dtype=float)
            coords = _ensure_closed(coords)
            cx, cy = _ring_centroid(coords)
            zonas.append(
                Zone(
                    alcaldia=str(r["alcaldia"]),
                    int2=int(r["int2"]),
                    area_m2=float(r["area_m2"]),
                    coords=coords,
                    centroid=(cx, cy),
                )
            )
        except Exception:
            # skip malformed rows
            continue
    if not zonas:
        raise ValueError("No Polygon geometries found in 'geo_shape'.")
    return zonas


@st.cache_data(show_spinner=False)
def load_zones_from_csv_path(path: Union[str, os.PathLike]) -> List[Zone]:
    with open(path, "rb") as f:
        return load_zones_from_csv_bytes(f.read())


def _sample_grid(nx=4, ny=4) -> List[Zone]:
    zonas: List[Zone] = []
    xmin, ymin, xmax, ymax = 0.0, 0.0, 4.0, 4.0
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    for j in range(ny):
        for i in range(nx):
            x0 = xmin + i * dx; y0 = ymin + j * dy
            coords = np.array([
                [x0, y0],[x0+dx, y0],[x0+dx, y0+dy],[x0, y0+dy],[x0, y0]
            ], dtype=float)
            zonas.append(
                Zone(
                    alcaldia=f"Z{j+1}-{i+1}",
                    int2=np.random.randint(1, 6),
                    area_m2=float(dx * dy * 1e6),
                    coords=coords,
                    centroid=_ring_centroid(coords),
                )
            )
    return zonas


# ---------------------- Model ----------------------

def generate_rain_series(zonas: List[Zone], frames: int, semilla_azar: int, coef_escorrentia: float) -> np.ndarray:
    """Return R (frames x N) effective rainfall after applying runoff coefficient."""
    rng = np.random.default_rng(semilla_azar)
    centroids = np.array([z.centroid for z in zonas])
    X = centroids[:, 0]; Y = centroids[:, 1]
    # normalize to [0,1]
    x = (X - X.min()) / (X.max() - X.min() + 1e-9)
    y = (Y - Y.min()) / (Y.max() - Y.min() + 1e-9)

    base = 0.55 + 0.25 * np.sin(2 * np.pi * (0.45 * x + 0.25 * y))           + 0.20 * np.cos(2 * np.pi * (0.25 * x - 0.35 * y))
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)

    R = []
    for t in range(frames):
        phase = 2 * np.pi * t / max(1, frames - 1)
        rt = 0.70 * base + 0.30 * (0.5 + 0.5 * np.sin(2 * np.pi * (x + y) + phase))
        rt += 0.05 * rng.standard_normal(len(zonas))
        rt = np.clip(rt, 0.0, None)
        R.append(rt)
    R = np.vstack(R) * float(coef_escorrentia)
    return R


def derive_capacity(
    zonas: List[Zone],
    R: np.ndarray,
    nivel_capacidad_base: float,
    efecto_riesgo: float,
    porcentaje_taponamiento: float
) -> np.ndarray:
    """Capacity per zone using area, risk (int2) and clogging porcentaje_taponamiento (0-1).
       efecto_riesgo ∈ [0,1]: 0=no risk effect, 1=full effect."""
    areas = np.array([z.area_m2 for z in zonas])
    a_norm = (areas - areas.min()) / (areas.max() - areas.min() + 1e-9)
    int2 = np.array([z.int2 for z in zonas])  # 1 .. 5

    # Risk multiplier: reduce capacity as risk increases. Tunable with efecto_riesgo.
    # At efecto_riesgo=1: capacity factor drops by ~0.6 from int2=1 to int2=5.
    risk_mult = 1.2 - 0.2 * efecto_riesgo * (int2 - 1)
    risk_mult = np.clip(risk_mult, 0.3, 1.2)

    mean_rain = float(R.mean())
    cap = nivel_capacidad_base * mean_rain * (0.9 + 0.2 * a_norm) * risk_mult

    # Clogging (taponamiento) — porcentaje_taponamiento ∈ [0,1] => reduce capacity linearly
    cap *= (1.0 - float(porcentaje_taponamiento))
    cap = np.clip(cap, 0.03 * mean_rain, 1.2 * mean_rain)
    return cap


def simulate_accumulation(R: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    T, N = R.shape
    A = np.zeros((T, N), dtype=float)
    for t in range(1, T):
        A[t] = np.maximum(0.0, A[t-1] + R[t] - capacity)
    return A


def pick_targets(values: np.ndarray, proporcion_intervencion: float) -> np.ndarray:
    n = len(values)
    k = max(1, int(np.ceil(proporcion_intervencion * n)))
    idxs = np.argsort(-values)[:k]
    mask = np.zeros(n, dtype=bool)
    mask[idxs] = True
    return mask


# ---------------------- Rendering ----------------------

def _render_boundaries_png(zonas: List[Zone], title: str = "") -> bytes:
    fig = plt.figure(figsize=(7, 7)); ax = plt.gca()
    for z in zonas:
        ax.add_patch(_make_patch(z.coords, facecolor='none', edgecolor='black', lw=0.25, alpha=1.0))
    ax.set_aspect('equal','box'); ax.autoscale_view(); ax.axis('off')
    if title: ax.set_title(title, pad=12)
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=220, bbox_inches='tight'); plt.close(fig)
    return buf.getvalue()


def _render_choropleth_png(zonas: List[Zone], values: np.ndarray, title: str, vmin=None, vmax=None, cmap_name: Optional[str] = 'Blues', cbar_label: Optional[str] = None) -> bytes:
    if vmin is None: vmin = float(values.min())
    if vmax is None: vmax = float(values.max() + 1e-9)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    fig = plt.figure(figsize=(7, 7)); ax = plt.gca()
    for z, val in zip(zonas, values):
        ax.add_patch(_make_patch(z.coords, facecolor=cmap(norm(val)), edgecolor='black', lw=0.2, alpha=1.0))
    ax.set_aspect('equal','box'); ax.autoscale_view(); ax.axis('off'); ax.set_title(title, pad=12)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=220, bbox_inches='tight'); plt.close(fig)
    return buf.getvalue()


# ---------------------- Animation helpers (GIF + MP4) ----------------------

def _build_side_by_side_anim(zonas, A0, A1, fps=8, overlay_opts=None):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    import numpy as np

    T, N = A0.shape
    vmax = float(max(A0.max(), A1.max()) + 1e-9)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap('Blues')

    # --- overlay options & constants ---
    area = np.array([z.area_m2 for z in zonas], dtype=float)
    depth_per_unit_m = 0.0
    items = []
    show = True
    minutes_per_frame = 0.0
    area_weighted = True
    if overlay_opts:
        depth_per_unit_m = max(0.0, float(overlay_opts.get("depth_per_unit_mm", 0.0))) / 1000.0
        items = list(overlay_opts.get("items", []))
        show = bool(overlay_opts.get("show", True))
        minutes_per_frame = float(overlay_opts.get("minutes_per_frame", 0.0))
        area_weighted = bool(overlay_opts.get("area_weighted", True))

    # Precompute per-frame stats
    if area_weighted:
        mean_baseline = (A0 * area).sum(axis=1) / area.sum()
        mean_desaz    = (A1 * area).sum(axis=1) / area.sum()
        mean_impact   = ((A0 - A1) * area).sum(axis=1) / area.sum()
    else:
        mean_baseline = A0.mean(axis=1)
        mean_desaz    = A1.mean(axis=1)
        mean_impact   = (A0 - A1).mean(axis=1)

    # Volumes (baseline/desazolve), and non-negative impact volume
    vol_baseline_rel = (A0 * area).sum(axis=1)         # unit·m²
    vol_desaz_rel    = (A1 * area).sum(axis=1)         # unit·m²
    vol_impact_rel   = (np.maximum(0.0, A0 - A1) * area).sum(axis=1)  # unit·m²

    if depth_per_unit_m > 0.0:
        vol_unit = "m³"
        vol_baseline = vol_baseline_rel * depth_per_unit_m
        vol_desaz    = vol_desaz_rel * depth_per_unit_m
        vol_impact   = vol_impact_rel * depth_per_unit_m
    else:
        vol_unit = "unit·m²"
        vol_baseline = vol_baseline_rel
        vol_desaz    = vol_desaz_rel
        vol_impact   = vol_impact_rel

    # Layout & patches
    all_x = np.concatenate([z.coords[:, 0] for z in zonas])
    dx = (float(all_x.max()) - float(all_x.min())) * 1.15
    shift = np.array([dx, 0.0])

    def _make_patch_local(coords, **kw):
        path = Path(coords, np.concatenate([[Path.MOVETO],
                                            np.full(len(coords)-1, Path.LINETO, dtype=Path.code_type)]))
        return PathPatch(path, **kw)

    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    patches_left, patches_right = [], []
    for z in zonas:
        pL = _make_patch_local(z.coords, facecolor=cmap(norm(0.0)), edgecolor='black', lw=0.2, alpha=1.0)
        pR = _make_patch_local(z.coords + shift, facecolor=cmap(norm(0.0)), edgecolor='black', lw=0.2, alpha=1.0)
        ax.add_patch(pL); ax.add_patch(pR)
        patches_left.append(pL); patches_right.append(pR)

    ax.set_aspect('equal','box'); ax.autoscale_view(); ax.axis('off')
    ax.text(0.25, 0.98, "Baseline",   transform=ax.transAxes, ha='center', va='top', fontsize=11)
    ax.text(0.75, 0.98, "Desazolve",  transform=ax.transAxes, ha='center', va='top', fontsize=11)

    overlay_txt = ax.text(
        0.01, 0.98, "", transform=ax.transAxes, va='top', ha='left', fontsize=10, family='monospace',
        bbox=dict(facecolor='white', alpha=0.65, boxstyle='round,pad=0.25')
    )

    def _overlay_line(name, value, unit=""):
        if unit:
            return f"{name}: {value:,.3f} {unit}"
        return f"{name}: {value:,.3f}"

    def _build_overlay_text(i: int) -> str:
        if not show:
            return ""
        lines = []
        if "Cuadro #" in items:
            lines.append(f"Frame: {i+1}/{T}")
        if "Progreso %" in items:
            lines.append(f"Progress: {100.0*(i+1)/T:5.1f}%")
        if "Tiempo transcurrido" in items:
            lines.append(f"t: {_format_elapsed_minutes(i * minutes_per_frame)}")
        if "Media base" in items:
            lines.append(_overlay_line("μ baseline", mean_baseline[i]))
        if "Media con desazolve" in items:
            lines.append(_overlay_line("μ desazolve", mean_desaz[i]))
        if "Media del impacto" in items:
            lines.append(_overlay_line("μ impacto", mean_impact[i]))
        if "Volumen total base" in items:
            lines.append(_overlay_line("V baseline", vol_baseline[i], vol_unit))
        if "Volumen total con desazolve" in items:
            lines.append(_overlay_line("V desazolve", vol_desaz[i], vol_unit))
        if "Volumen total del impacto (≥0)" in items:
            lines.append(_overlay_line("V impacto (≥0)", vol_impact[i], vol_unit))
        return "\n".join(lines)

    def update(i):
        for p, v in zip(patches_left,  A0[i]): p.set_facecolor(cmap(norm(v)))
        for p, v in zip(patches_right, A1[i]): p.set_facecolor(cmap(norm(v)))
        overlay_txt.set_text(_build_overlay_text(i))
        return patches_left + patches_right + [overlay_txt]

    anim = FuncAnimation(fig, update, frames=T, interval=int(1000/fps), blit=False)
    return fig, anim




def _save_anim_to_gif_bytes(fig, anim, fps=8, dpi=130) -> bytes:
    """Robust GIF saver via PillowWriter."""
    import tempfile, os
    from matplotlib.animation import PillowWriter

    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        anim.save(tmp_path, writer=PillowWriter(fps=fps), dpi=dpi)  # moderate DPI for compatibility
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try: os.remove(tmp_path)
        except OSError: pass


def _save_anim_to_mp4_bytes(fig, anim, fps=8, dpi=130, bitrate=1800):
    """MP4 (H.264) fallback for browsers/viewers that don't animate GIFs well."""
    import tempfile, os
    try:
        from matplotlib.animation import FFMpegWriter
    except Exception:
        return None  # ffmpeg bindings not present

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=bitrate)
        anim.save(tmp_path, writer=writer, dpi=dpi)
        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception:
        # ffmpeg not installed or failed – just skip MP4
        return None
    finally:
        try: os.remove(tmp_path)
        except OSError: pass

import streamlit.components.v1 as components

def _embed_anim_jshtml(fig, anim, height_px: Union[int, None] = None) -> bool:
    """
    Embed the animation using Matplotlib's JS HTML player.
    Returns True if embedded, False on failure.
    """
    try:
        html = anim.to_jshtml()  # pure-Python, no ffmpeg
        if height_px is None:
            # approximate CSS pixels from figure size
            height_px = int(fig.get_size_inches()[1] * 110) + 100
        components.html(html, height=height_px, scrolling=False)
        return True
    except Exception:
        return False
def _format_elapsed_minutes(total_minutes: float) -> str:
    h = int(total_minutes // 60)
    m = int(round(total_minutes % 60))
    return f"{h}h {m:02d}m" if h > 0 else f"{m}m"

# ---------------------- UI ----------------------

st.title("CDMX — Simulador de Lluvia y Desazolve con Enjambre")

with st.sidebar:
    st.header("1) Datos")
    data_src = st.radio(
        "Elige la fuente de datos",
        ["Atlas CDMX incluido (predeterminado)", "Subir CSV", "Cuadrícula de demostración"],
        index=0,
        help="Para incluir el atlas, colócalo en data/atlas-de-riesgo-inundaciones.csv"
    )
    up = None
    use_sample = False
    if data_src == "Subir CSV":
        up = st.file_uploader("Subir atlas-de-riesgo-inundaciones.csv", type=["csv"])
    elif data_src == "Cuadrícula de demostración":
        use_sample = True

    st.header("2) Parámetros")
    frames = st.slider("Frames (horizonte de animación)", 10, 80, 28, step=1)
    semilla_azar = st.number_input("Semilla aleatoria", min_value=0, value=15, step=1)

    st.markdown("**Lluvia y escorrentía (mm/h)**")
    coef_escorrentia = st.slider("Coeficiente de escorrentía C_r (adimensional)", 0.2, 1.2, 1.0, step=0.05)

    st.markdown("**Capacidad de drenaje (mm/h)**")
    nivel_capacidad_base = st.slider("Nivel de capacidad base (relativo, mm/h)", 0.20, 1.00, 0.55, step=0.05)
    efecto_riesgo = st.slider("Efecto del riesgo (int2) sobre la capacidad (relativo)", 0.0, 1.0, 1.0, step=0.05)
    porcentaje_taponamiento = st.slider("Porcentaje de taponamiento porcentaje_taponamiento (%)", 0, 80, 20, step=5, help="Global baseline taponamiento (%)")
    reduccion_taponamiento = st.slider("Reducción de taponamiento con desazolve (%)", 0, 80, 20, step=5, help="Reduce porcentaje_taponamiento by this many percentage points in intervened zonas")

    st.markdown("**Focalización de desazolve (barquito)**")
    proporcion_intervencion = st.slider("Proporción intervenida (proporcion_intervencion, % de zonas)", 0.0, 0.9, 0.35, step=0.05)
    delta_capacidad = st.slider("Incremento adicional de capacidad δ (zonas intervenidas, %)", 0.0, 1.5, 0.45, step=0.05)

    generar_gif = st.checkbox("También generar GIF/MP4 comparativo", value=True)
    st.header("3) Superposición en video")
    show_overlay = st.checkbox("Mostrar superposición en video", value=True)
    st.header("4) Unidades y agregación")
    depth_per_unit_mm = st.number_input(
        "Profundidad por unidad del modelo (mm)", min_value=0.0, max_value=1000.0, value=0.0, step=0.5,
        help="Coloca un valor > 0 para convertir la profundidad del modelo a milímetros y reportar volúmenes en m³. Deja 0 para unidades relativas (u·m²)."
    )
    area_weighted = st.checkbox("Usar medias ponderadas por área (μ)", value=True)

    overlay_items = st.multiselect(
    "Elementos a superponer",
    [
        "Cuadro #", "Progreso %", "Tiempo transcurrido",
        "Media base", "Media con desazolve", "Media del impacto",
        "Volumen total base", "Volumen total con desazolve", "Volumen total del impacto (≥0)"
    ],
    default=["Cuadro #", "Tiempo transcurrido", "Media del impacto", "Volumen total del impacto (≥0)"]
)



    minutes_per_frame = st.slider("Minutos por cuadro (Δt)", 0.0, 180.0, 10.0, step=5.0,
                                  help="Se usa para calcular la etiqueta de 'Tiempo transcurrido'")
    
    ejecutar = st.button("Ejecutar simulación", type="primary")


# Load zonas
zonas: List[Zone] = []
try:
    if use_sample:
        zonas = _sample_grid(6, 6)
    elif up is not None:
        zonas = load_zones_from_csv_bytes(up.getvalue())
    else:
        if DEFAULT_ATLAS_PATH.exists():
            zonas = load_zones_from_csv_path(DEFAULT_ATLAS_PATH)
            st.caption(f"Cargando datos base del Atlas de Riesgo {DEFAULT_ATLAS_PATH}")
        else:
            st.info("Carga un CSV, o elige 'Demo'.")
except FileNotFoundError:
    st.error(f"no se encontró archivo default {DEFAULT_ATLAS_PATH}.")
except Exception as e:
    st.error(f"Error loading data: {e}")


if ejecutar and zonas:
    # --------- Simulate ----------
    R = generate_rain_series(zonas, frames=frames, semilla_azar=semilla_azar, coef_escorrentia=coef_escorrentia)

    cap_baseline = derive_capacity(
        zonas, R,
        nivel_capacidad_base=nivel_capacidad_base,
        efecto_riesgo=efecto_riesgo,
        porcentaje_taponamiento=porcentaje_taponamiento / 100.0
    )

    A0 = simulate_accumulation(R, cap_baseline)          # baseline
    A0f = A0[-1]

    # Interventions
    mask = pick_targets(A0f, proporcion_intervencion=proporcion_intervencion)
    # Effective porcentaje_taponamiento after desazolve (apply reduction only in intervened zonas)
    p_tap_vec = np.full(len(zonas), porcentaje_taponamiento / 100.0)
    p_tap_vec[mask] = np.clip(p_tap_vec[mask] - (reduccion_taponamiento / 100.0), 0.0, 1.0)

    cap_desaz = derive_capacity(
        zonas, R,
        nivel_capacidad_base=nivel_capacidad_base * (1.0 + delta_capacidad),   # multiplicative boost δ
        efecto_riesgo=efecto_riesgo,
        porcentaje_taponamiento=0.0  # we pass 0 here and use p_tap_vec multiplicatively below
    )
    # Apply zone-specific porcentaje_taponamiento after desazolve
    cap_desaz *= (1.0 - p_tap_vec)

    A1 = simulate_accumulation(R, cap_desaz)            # desazolve
    A1f = A1[-1]
    impact = np.maximum(0.0, A0f - A1f)

    # --------- Render ----------
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.caption("Boundaries")
        st.image(_render_boundaries_png(zonas), use_column_width=True)
    vmin, vmax = 0.0, float(max(A0f.max(), A1f.max()) + 1e-9)
    with colB:
        st.caption("Línea base — Profundidad de inundación acumulada")
        png = _render_choropleth_png(zonas, A0f, "Control: Profundidad final", vmin=vmin, vmax=vmax, cmap_name='Blues', cbar_label=("Profundidad [mm]" if depth_per_unit_mm>0 else "Profundidad [u]"))
        st.image(png, use_column_width=True)
        st.download_button("Descargar PNG", data=png, file_name="baseline_final.png", mime="image/png")
    with colC:
        st.caption("Desazolve — Profundidad de inundación acumulada")
        png = _render_choropleth_png(zonas, A1f, "Desazolve: Profundidad final", vmin=vmin, vmax=vmax, cmap_name='Blues', cbar_label=("Profundidad [mm]" if depth_per_unit_mm>0 else "Profundidad [u]"))
        st.image(png, use_column_width=True)
        st.download_button("Descargar PNG", data=png, file_name="desazolve_final.png", mime="image/png")
    with colD:
        st.caption("Impacto — Reducción (Base − Desazolve)")
        png = _render_choropleth_png(zonas, impact, "Impacto (Base − Desazolve): Reducción", vmin=0.0, vmax=float(impact.max()+1e-9), cmap_name='Greens', cbar_label=("Reducción [mm]" if depth_per_unit_mm>0 else "Reducción [u]"))
        st.image(png, use_column_width=True)
        st.download_button("Descargar PNG", data=png, file_name="impact_reduction.png", mime="image/png")

    # Summary
    
    st.subheader("Resumen con unidades y mejoras")
    df = pd.DataFrame({
        "alcaldia": [z.alcaldia for z in zonas],
        "baseline_final": A0f,
        "desazolve_final": A1f,
        "impact": impact,
        "intervened": mask,
        "area_m2": [z.area_m2 for z in zonas],
    })
    # Conversión de unidades
    if depth_per_unit_mm > 0:
        df["baseline_mm"] = df["baseline_final"] * depth_per_unit_mm
        df["desazolve_mm"] = df["desazolve_final"] * depth_per_unit_mm
        df["impact_mm"] = df["impact"] * depth_per_unit_mm
        df["impact_m"] = df["impact_mm"] / 1000.0
        df["vol_impact_m3"] = df["impact_m"] * df["area_m2"]
        prof_base_label = "Profundidad media — base [mm]"
        prof_des_label  = "Profundidad media — desazolve [mm]"
        red_med_label   = "Reducción media [mm]"
        vol_tot_label   = "Volumen total evitado (≥0) [m³]"
    else:
        df["baseline_u"] = df["baseline_final"]
        df["desazolve_u"] = df["desazolve_final"]
        df["impact_u"] = df["impact"]
        df["vol_impact_um2"] = df["impact_u"] * df["area_m2"]
        prof_base_label = "Profundidad media — base [u]"
        prof_des_label  = "Profundidad media — desazolve [u]"
        red_med_label   = "Reducción media [u]"
        vol_tot_label   = "Volumen total evitado (≥0) [u·m²]"
    
    def _percent(a, b):
        return (a / b * 100.0) if b != 0 else 0.0
    
    grupos = []
    for alc, g in df.groupby("alcaldia"):
        if depth_per_unit_mm > 0:
            base_mean = g["baseline_mm"].mean()
            des_mean  = g["desazolve_mm"].mean()
            red_mean  = g["impact_mm"].mean()
            vol_tot   = g["vol_impact_m3"].sum()
        else:
            base_mean = g["baseline_u"].mean()
            des_mean  = g["desazolve_u"].mean()
            red_mean  = g["impact_u"].mean()
            vol_tot   = g["vol_impact_um2"].sum()
        pct = _percent(red_mean, base_mean)
        grupos.append({
            "Alcaldía": alc,
            "Zonas (n)": len(g),
            prof_base_label: base_mean,
            prof_des_label: des_mean,
            red_med_label: red_mean,
            "Reducción media [%]": pct,
            vol_tot_label: vol_tot,
        })
    resumen = pd.DataFrame(grupos).sort_values(vol_tot_label, ascending=False)
    
    def _style_green(value: float) -> str:
        try:
            return 'color: green' if float(value) > 0 else ''
        except Exception:
            return ''

    styler = (
        resumen.style
        .format({
            prof_base_label: '{:,.2f}',
            prof_des_label: '{:,.2f}',
            red_med_label: '{:,.2f}',
            "Reducción media [%]": '{:,.2f}',
            vol_tot_label: '{:,.2f}',
        })
        .map(_style_green, subset=[red_med_label, "Reducción media [%]", vol_tot_label])
    )

    st.dataframe(styler, use_container_width=True)
    
    csv_bytes = resumen.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar CSV del resumen", data=csv_bytes, file_name="resumen_con_unidades.csv", mime="text/csv")
    
        
    # Optional side-by-side animation — mostrar como VIDEO en la app
    if generar_gif:
        st.subheader("Relleno lado a lado: Base (izquierda) vs Desazolve (derecha)")
    
        overlay_opts = {
            "show": show_overlay,
            "items": overlay_items,
            "minutes_per_frame": minutes_per_frame,
            "depth_per_unit_mm": depth_per_unit_mm,   # NEW
            "area_weighted": area_weighted,           # NEW
        }
        fig, anim = _build_side_by_side_anim(zonas, A0, A1, fps=8, overlay_opts=overlay_opts)
    
    
    
        # 1) Try MP4 first (played by st.video)
        mp4_bytes = _save_anim_to_mp4_bytes(fig, anim, fps=8, dpi=130, bitrate=1800)
        played = False
        if mp4_bytes:
            st.video(mp4_bytes)
            st.download_button("Descargar MP4", data=mp4_bytes,
                            file_name="baseline_vs_desazolve.mp4", mime="video/mp4")
            played = True
    
        # 2) If MP4 fails (no ffmpeg), embed JSHTML player (no external deps)
        if not played:
            if _embed_anim_jshtml(fig, anim):
                st.caption("Playing inline via HTML (no ffmpeg).")
                played = True
    
        # 3) Always provide a GIF as a download for sharing (but don't try to play it inline)
        gif_bytes = _save_anim_to_gif_bytes(fig, anim, fps=8, dpi=130)
        st.download_button("Descargar GIF (compatibilidad)", data=gif_bytes,
                        file_name="baseline_vs_desazolve.gif", mime="image/gif")
    
        if not played:
            st.warning("No se pudo reproducir el video en línea. Descarga el MP4 o GIF de arriba.")
elif ejecutar:
    st.warning("Carga un dataset válido antes de ejecutar la simulación.")

# Notes
st.markdown("""
**Notas**

- El llenado es un modelo conceptual de “cubo” para comunicación y priorización.
- Para convertir profundidades relativas a **mm** o **m³**, calibra las unidades de lluvia con pluviómetros/aforos.
- Parámetros que puedes ajustar aquí:
  - Serie de lluvia: cuadros, semilla, **coeficiente de escorrentía (C_r)**.
  - Capacidad de drenaje: nivel base, **efecto de riesgo (int2)**, **taponamiento (porcentaje_taponamiento)**.
  - Intervenciones: proporción de zonas (**proporcion_intervencion**) y **δ** (+ capacidad), además de **reducción de porcentaje_taponamiento** en zonas intervenidas.
""")
