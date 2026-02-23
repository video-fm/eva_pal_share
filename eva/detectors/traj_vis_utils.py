"""
traj_vis_utils.py
-----------------
Draw smooth trajectory overlays on images with colour-gradient lines.

Usage:
    from eva.detectors.traj_vis_utils import add_arrow
    result = add_arrow("photo.png", [(100, 200), (150, 180), (250, 220), (300, 200)])
    result.save("output.png")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import splprep, splev


def get_image_resized(img_path: str, max_size: int = 1024) -> Image.Image:
    """Load image and resize if needed for API."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    return img


def add_arrow(
    image: Union[str, np.ndarray, Image.Image],
    points: Sequence[Tuple[float, float]],
    color: Union[str, Tuple[int, ...]] = "red",
    line_width: int = 2,
    arrow_head_length: float = 20.0,
    arrow_head_angle: float = 30.0,
    num_interpolated: int = 300,
    smoothing: float = 0.0,
) -> Image.Image:
    """Draw a smooth red-to-pink gradient trajectory on *image* through *points*.

    Parameters
    ----------
    image : str | np.ndarray | PIL.Image.Image
        The background image. A file path, numpy array (H×W×C, uint8), or
        PIL Image.
    points : sequence of (x, y) tuples
        At least 2 control points that define the trajectory path.
    color, arrow_head_length, arrow_head_angle
        Kept for backwards compatibility; ignored.
    line_width : int
        Width of the trajectory line in pixels.
    num_interpolated : int
        Number of sample points along the fitted spline (higher = smoother).
    smoothing : float
        Spline smoothing factor passed to ``scipy.interpolate.splprep``.
        0 means the spline passes exactly through the given points.

    Returns
    -------
    PIL.Image.Image
        A copy of the input image with the trajectory drawn on top.
    """
    cfg = TraceOverlayConfig(
        future_color=(255, 0, 0),
        future_color_end=(255, 105, 180),
        future_thickness=line_width,
        future_outline_thickness=line_width + 3,
        future_outline_color=(0, 0, 0),
    )
    return add_trace_overlay(
        image, points, current_index=0, config=cfg,
        num_interpolated=num_interpolated, smoothing=smoothing,
    )


@dataclass
class TraceOverlayConfig:
    """All visual parameters for trace overlay rendering."""

    max_shift: float = 20.0

    horizon: int = 0
    show_past: bool = False
    past_horizon: int = 0

    future_color: Tuple[int, int, int] = (255, 0, 255)
    future_color_end: Optional[Tuple[int, int, int]] = None
    future_outline_color: Tuple[int, int, int] = (0, 0, 0)
    future_thickness: int = 1
    future_outline_thickness: int = 0

    past_color: Tuple[int, int, int] = (180, 180, 180)
    past_color_end: Optional[Tuple[int, int, int]] = None
    past_outline_color: Tuple[int, int, int] = (0, 0, 0)
    past_thickness: int = 1
    past_outline_thickness: int = 0

    current_dot_radius: int = 0
    current_dot_color: Tuple[int, int, int] = (255, 255, 0)
    current_dot_outline_color: Tuple[int, int, int] = (0, 0, 0)
    current_dot_outline_thickness: int = 0

    use_alpha: bool = False
    alpha: float = 1.0

    dashed_future: bool = False
    dash_len: int = 10
    gap_len: int = 6

    arrow_mode: Literal["end_only", "multiple"] = "end_only"
    arrow_count: int = 0
    arrow_size: int = 12
    arrow_thickness: int = 1
    arrow_color: Tuple[int, int, int] = (0, 255, 255)
    arrow_outline_color: Tuple[int, int, int] = (0, 0, 0)
    arrow_outline_thickness: int = 0

    tick_marks: bool = False
    tick_every: int = 5
    tick_radius: int = 2
    tick_color: Tuple[int, int, int] = (255, 255, 255)
    tick_outline_color: Tuple[int, int, int] = (0, 0, 0)


def _lerp_color(
    c0: Tuple[int, int, int],
    c1: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    """Linearly interpolate between two RGB colours. *t* in [0, 1]."""
    return (
        round(c0[0] + (c1[0] - c0[0]) * t),
        round(c0[1] + (c1[1] - c0[1]) * t),
        round(c0[2] + (c1[2] - c0[2]) * t),
    )


def _draw_dashed_polyline_cv(
    img: np.ndarray,
    pts: list[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int,
    dash_len: int,
    gap_len: int,
    color_end: Optional[Tuple[int, int, int]] = None,
) -> None:
    if len(pts) < 2:
        return

    total_len = 0.0
    if color_end is not None:
        for i in range(len(pts) - 1):
            total_len += math.hypot(
                float(pts[i + 1][0]) - float(pts[i][0]),
                float(pts[i + 1][1]) - float(pts[i][1]),
            )
        if total_len < 1e-6:
            return

    cycle = dash_len + gap_len
    accum = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = float(pts[i][0]), float(pts[i][1])
        x1, y1 = float(pts[i + 1][0]), float(pts[i + 1][1])
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len < 1e-6:
            continue
        dx, dy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
        d = 0.0
        while d < seg_len:
            phase = (accum + d) % cycle
            if phase < dash_len:
                dash_remaining = dash_len - phase
                end_d = min(d + dash_remaining, seg_len)
                p0 = (round(x0 + dx * d), round(y0 + dy * d))
                p1 = (round(x0 + dx * end_d), round(y0 + dy * end_d))
                c = _lerp_color(color, color_end, (accum + d) / total_len) if color_end is not None else color
                cv2.line(img, p0, p1, c, thickness, cv2.LINE_AA)
                d = end_d
            else:
                d += cycle - phase
        accum += seg_len


def _draw_polyline_with_outline(
    img: np.ndarray,
    pts: list[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int,
    outline_color: Tuple[int, int, int],
    outline_thickness: int,
    *,
    color_end: Optional[Tuple[int, int, int]] = None,
    is_dashed: bool = False,
    dash_len: int = 10,
    gap_len: int = 6,
    alpha: float = 1.0,
    shift: int = 0,
) -> np.ndarray:
    if len(pts) < 2:
        return img
    overlay = img.copy() if alpha < 1.0 else img
    if is_dashed:
        if outline_thickness > 0:
            _draw_dashed_polyline_cv(overlay, pts, outline_color, outline_thickness, dash_len, gap_len)
        _draw_dashed_polyline_cv(overlay, pts, color, thickness, dash_len, gap_len, color_end=color_end)
    elif color_end is not None:
        # Gradient: draw outline as solid first, then per-segment gradient on top
        np_pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        if outline_thickness > 0:
            cv2.polylines(overlay, [np_pts], False, outline_color, outline_thickness, cv2.LINE_AA, shift=shift)
        n = len(pts) - 1
        for i in range(n):
            t = i / max(n - 1, 1)
            seg_color = _lerp_color(color, color_end, t)
            seg = np.array([pts[i], pts[i + 1]], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [seg], False, seg_color, thickness, cv2.LINE_AA, shift=shift)
    else:
        np_pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        if outline_thickness > 0:
            cv2.polylines(overlay, [np_pts], False, outline_color, outline_thickness, cv2.LINE_AA, shift=shift)
        cv2.polylines(overlay, [np_pts], False, color, thickness, cv2.LINE_AA, shift=shift)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)
    return img


def _draw_circle_with_outline(
    img: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    outline_color: Tuple[int, int, int],
    outline_thickness: int,
    alpha: float = 1.0,
    shift: int = 0,
) -> np.ndarray:
    if radius <= 0:
        return img
    scale = 1 << shift
    overlay = img.copy() if alpha < 1.0 else img
    if outline_thickness > 0:
        cv2.circle(
            overlay,
            center,
            radius + outline_thickness * scale // 2,
            outline_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
            shift=shift,
        )
    cv2.circle(overlay, center, radius, color, thickness=-1, lineType=cv2.LINE_AA, shift=shift)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)
    return img


def _draw_arrows_with_outline(
    img: np.ndarray,
    pts: list[Tuple[int, int]],
    indices: list[int],
    color: Tuple[int, int, int],
    outline_color: Tuple[int, int, int],
    thickness: int,
    outline_thickness: int,
    tip_length_px: int,
    alpha: float = 1.0,
    shift: int = 0,
) -> np.ndarray:
    if len(pts) < 2 or not indices:
        return img
    overlay = img.copy() if alpha < 1.0 else img
    for idx in indices:
        if idx < 0 or idx + 1 >= len(pts):
            continue
        p0, p1 = pts[idx], pts[idx + 1]
        seg_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if seg_len < 1e-3:
            continue
        scale = 1 << shift
        tip_ratio = min(tip_length_px * scale / seg_len, 0.5)
        if outline_thickness > 0:
            cv2.arrowedLine(overlay, p0, p1, outline_color, outline_thickness, cv2.LINE_AA, shift=shift, tipLength=tip_ratio)
        cv2.arrowedLine(overlay, p0, p1, color, thickness, cv2.LINE_AA, shift=shift, tipLength=tip_ratio)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)
    return img


def _arrow_indices(future_len: int, cfg: TraceOverlayConfig) -> list[int]:
    if future_len < 2 or cfg.arrow_count == 0:
        return []
    if cfg.arrow_mode == "end_only":
        return [future_len - 2]
    count = min(cfg.arrow_count, future_len - 1)
    if count <= 0:
        return []
    step = (future_len - 1) / (count + 1)
    return [round(step * (i + 1)) for i in range(count)]


def _annotate_frame(
    frame_rgb: np.ndarray,
    pts: list[Tuple[int, int]],
    t: int,
    cfg: TraceOverlayConfig,
    shift: int = 0,
) -> np.ndarray:
    """Draw all trace overlays for frame index *t* on *frame_rgb* (uint8, RGB).

    Coordinates in *pts* are fixed-point with *shift* fractional bits.
    """
    n = len(pts)
    if n == 0:
        return frame_rgb

    t = max(0, min(t, n - 1))
    alpha = cfg.alpha if cfg.use_alpha else 1.0

    future_end = n if cfg.horizon <= 0 else min(t + 1 + cfg.horizon, n)
    future_pts = pts[t:future_end]

    past_pts: list[Tuple[int, int]] = []
    if cfg.show_past and t > 0:
        past_start = 0 if cfg.past_horizon <= 0 else max(0, t - cfg.past_horizon)
        past_pts = pts[past_start : t + 1]

    scale = 1 << shift

    if len(past_pts) >= 2:
        frame_rgb = _draw_polyline_with_outline(
            frame_rgb, past_pts,
            color=cfg.past_color, thickness=cfg.past_thickness,
            outline_color=cfg.past_outline_color, outline_thickness=cfg.past_outline_thickness,
            color_end=cfg.past_color_end,
            alpha=alpha, shift=shift,
        )

    if len(future_pts) >= 2:
        frame_rgb = _draw_polyline_with_outline(
            frame_rgb, future_pts,
            color=cfg.future_color, thickness=cfg.future_thickness,
            outline_color=cfg.future_outline_color, outline_thickness=cfg.future_outline_thickness,
            color_end=cfg.future_color_end,
            is_dashed=cfg.dashed_future, dash_len=cfg.dash_len, gap_len=cfg.gap_len,
            alpha=alpha, shift=shift,
        )

    if cfg.tick_marks and len(future_pts) >= 2:
        for i in range(1, len(future_pts)):
            if i % cfg.tick_every == 0:
                frame_rgb = _draw_circle_with_outline(
                    frame_rgb, future_pts[i], cfg.tick_radius * scale,
                    color=cfg.tick_color, outline_color=cfg.tick_outline_color,
                    outline_thickness=1, alpha=alpha, shift=shift,
                )

    if len(future_pts) >= 2:
        arrow_idxs = _arrow_indices(len(future_pts), cfg)
        frame_rgb = _draw_arrows_with_outline(
            frame_rgb, future_pts, arrow_idxs,
            color=cfg.arrow_color, outline_color=cfg.arrow_outline_color,
            thickness=cfg.arrow_thickness, outline_thickness=cfg.arrow_outline_thickness,
            tip_length_px=cfg.arrow_size, alpha=alpha, shift=shift,
        )

    return _draw_circle_with_outline(
        frame_rgb, pts[t], cfg.current_dot_radius * scale,
        color=cfg.current_dot_color, outline_color=cfg.current_dot_outline_color,
        outline_thickness=cfg.current_dot_outline_thickness, alpha=alpha, shift=shift,
    )


def add_trace_overlay(
    image: Union[str, np.ndarray, Image.Image],
    points: Sequence[Tuple[float, float]],
    current_index: int = 0,
    config: TraceOverlayConfig | None = None,
    num_interpolated: int = 300,
    smoothing: float = 0.0,
) -> Image.Image:
    """Draw a trace overlay on *image* showing the trajectory through *points*.

    Parameters
    ----------
    image : str | np.ndarray | PIL.Image.Image
        The background image.  A file path, numpy array (H×W×C, uint8), or
        PIL Image.
    points : sequence of (x, y) tuples
        Trajectory coordinates in pixel space.
    current_index : int
        Index into *points* representing the current position.  The past
        trajectory is drawn behind this index and the future trajectory
        ahead of it.  When *num_interpolated* > len(points), this is
        automatically rescaled to the corresponding position in the
        upsampled curve.
    config : TraceOverlayConfig | None
        Visual parameters.  ``None`` uses all defaults.
    num_interpolated : int
        Number of sample points along a fitted spline.  Higher values
        produce smoother curves and finer colour gradients.  Set to 0
        to skip interpolation and use the raw input points.
    smoothing : float
        Spline smoothing factor passed to ``scipy.interpolate.splprep``.
        0 means the spline passes exactly through the given points.

    Returns
    -------
    PIL.Image.Image
        A copy of the input image with the trace overlay drawn on top.
    """
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.copy().convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if config is None:
        config = TraceOverlayConfig()

    pts_arr = np.asarray(points, dtype=float)
    if pts_arr.ndim != 2 or pts_arr.shape[1] != 2:
        raise ValueError("points must be an (N, 2) array of (x, y) coordinates")
    if len(pts_arr) < 2:
        raise ValueError("Need at least 2 points to draw a trace")

    # Spline-interpolate for smooth curves and fine-grained colour gradients
    n_raw = len(pts_arr)
    if num_interpolated > 0 and n_raw >= 2:
        x, y = pts_arr[:, 0], pts_arr[:, 1]
        if n_raw == 2:
            t_fine = np.linspace(0, 1, num_interpolated)
            xs = x[0] + (x[1] - x[0]) * t_fine
            ys = y[0] + (y[1] - y[0]) * t_fine
        else:
            k = min(3, n_raw - 1)
            tck, _ = splprep([x, y], s=smoothing, k=k)
            t_fine = np.linspace(0, 1, num_interpolated)
            xs, ys = splev(t_fine, tck)
        pts_arr = np.column_stack([xs, ys])
        # Rescale current_index to the interpolated curve
        current_index = round(current_index / max(n_raw - 1, 1) * (len(pts_arr) - 1))

    shift = 4
    scale = 1 << shift
    fixed_pts = [(round(cx * scale), round(cy * scale)) for cx, cy in pts_arr.tolist()]

    frame_rgb = np.array(img, dtype=np.uint8).copy()
    frame_rgb = _annotate_frame(frame_rgb, fixed_pts, current_index, config, shift=shift)

    return Image.fromarray(frame_rgb)


# ---------------------------------------------------------------------------
# Quick demos / CLI
# ---------------------------------------------------------------------------
def _demo_trace_overlay(image_path: str, points: list[Tuple[float, float]], output_dir: str) -> None:
    """Generate several trace-overlay examples with different configs."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    mid = len(points) // 2

    RED = (255, 0, 0)
    PINK = (255, 105, 180)

    configs: dict[str, tuple[int, TraceOverlayConfig]] = {
        "01_red_to_pink": (
            0,
            TraceOverlayConfig(
                future_color=RED,
                future_color_end=PINK,
                future_thickness=3,
            ),
        ),
        "02_red_to_pink_outlined": (
            0,
            TraceOverlayConfig(
                future_color=RED,
                future_color_end=PINK,
                future_thickness=2,
                future_outline_thickness=5,
                future_outline_color=(0, 0, 0),
            ),
        ),
        "03_red_to_pink_with_past": (
            mid,
            TraceOverlayConfig(
                future_color=RED,
                future_color_end=PINK,
                future_thickness=3,
                show_past=True,
                past_color=(180, 180, 180),
                past_color_end=(100, 100, 100),
                past_thickness=2,
            ),
        ),
        "04_red_to_pink_dashed": (
            0,
            TraceOverlayConfig(
                future_color=RED,
                future_color_end=PINK,
                future_thickness=2,
                dashed_future=True,
                dash_len=10,
                gap_len=5,
            ),
        ),
        "05_red_to_pink_arrow": (
            0,
            TraceOverlayConfig(
                future_color=RED,
                future_color_end=PINK,
                future_thickness=3,
                arrow_count=1,
                arrow_size=14,
                arrow_thickness=2,
                arrow_color=PINK,
            ),
        ),
        "06_red_to_pink_dot_and_past": (
            mid,
            TraceOverlayConfig(
                future_color=RED,
                future_color_end=PINK,
                future_thickness=3,
                show_past=True,
                past_color=PINK,
                past_color_end=RED,
                past_thickness=2,
                current_dot_radius=6,
                current_dot_color=(255, 255, 0),
                current_dot_outline_color=(0, 0, 0),
                current_dot_outline_thickness=2,
            ),
        ),
        "07_red_to_pink_full": (
            mid,
            TraceOverlayConfig(
                future_color=RED,
                future_color_end=PINK,
                future_thickness=3,
                future_outline_thickness=5,
                future_outline_color=(0, 0, 0),
                show_past=True,
                past_color=(180, 180, 180),
                past_color_end=(100, 100, 100),
                past_thickness=2,
                past_outline_thickness=4,
                past_outline_color=(0, 0, 0),
                arrow_count=1,
                arrow_size=14,
                arrow_thickness=2,
                arrow_color=PINK,
                current_dot_radius=5,
                current_dot_color=(255, 255, 0),
                current_dot_outline_color=(0, 0, 0),
                current_dot_outline_thickness=2,
            ),
        ),
    }

    for name, (idx, cfg) in configs.items():
        out_path = os.path.join(output_dir, f"trace_{name}.png")
        result = add_trace_overlay(image_path, points, current_index=idx, config=cfg)
        result.save(out_path)
        print(f"  {out_path}")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Draw arrows or trace overlays on images.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- arrow subcommand ---
    p_arrow = sub.add_parser("arrow", help="Draw a red-to-pink gradient trajectory")
    p_arrow.add_argument("image", help="Path to the input image")
    p_arrow.add_argument(
        "points",
        help='Semicolon-separated points, e.g. "100,200;150,180;300,200"',
    )
    p_arrow.add_argument("-o", "--output", default="output.png")
    p_arrow.add_argument("--width", type=int, default=3, help="Line width")

    # --- trace subcommand ---
    p_trace = sub.add_parser("trace", help="Draw a trace overlay")
    p_trace.add_argument("image", help="Path to the input image")
    p_trace.add_argument(
        "points",
        help='Semicolon-separated points, e.g. "100,200;150,180;300,200"',
    )
    p_trace.add_argument("-o", "--output", default="trace_output.png")
    p_trace.add_argument("--index", type=int, default=0, help="Current-position index")
    p_trace.add_argument("--show-past", action="store_true")
    p_trace.add_argument("--dashed", action="store_true")
    p_trace.add_argument("--arrows", type=int, default=0, help="Number of arrows")
    p_trace.add_argument("--dot-radius", type=int, default=0)
    p_trace.add_argument("--thickness", type=int, default=2)

    # --- demo subcommand ---
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _default_demo_image = "/home/franka/eva_jiani/data/test_traj/2026-02-20_11-41-51/frame_step_000.jpg"
    _default_demo_points = "541.696,396.864;560,375;580.608,351.36"

    p_demo = sub.add_parser("demo", help="Generate gallery of trace-overlay styles")
    p_demo.add_argument("image", nargs="?", default=_default_demo_image, help="Path to the input image")
    p_demo.add_argument(
        "points",
        nargs="?",
        default=_default_demo_points,
        help='Semicolon-separated points, e.g. "100,200;150,180;300,200"',
    )
    p_demo.add_argument("-o", "--output-dir", default="trace_demos")

    args = parser.parse_args()

    if args.command == "arrow":
        pts = [tuple(map(float, p.split(","))) for p in args.points.split(";")]
        result = add_arrow(args.image, pts, line_width=args.width)
        result.save(args.output)
        print(f"Saved to {args.output}")

    elif args.command == "trace":
        pts = [tuple(map(float, p.split(","))) for p in args.points.split(";")]
        cfg = TraceOverlayConfig(
            show_past=args.show_past,
            dashed_future=args.dashed,
            arrow_count=args.arrows,
            current_dot_radius=args.dot_radius,
            future_thickness=args.thickness,
            past_thickness=args.thickness,
        )
        result = add_trace_overlay(args.image, pts, current_index=args.index, config=cfg)
        result.save(args.output)
        print(f"Saved to {args.output}")

    elif args.command == "demo":
        pts = [tuple(map(float, p.split(","))) for p in args.points.split(";")]
        print(f"Generating demos in {args.output_dir}/")
        _demo_trace_overlay(args.image, pts, args.output_dir)
        print("Done.")

    else:
        parser.print_help()
