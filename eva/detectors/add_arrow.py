"""
add_arrow.py
------------
Draw a smooth curved arrow on an image, fitted through a sequence of points.

Usage:
    from add_arrow import add_arrow
    result = add_arrow("photo.png", [(100, 200), (150, 180), (250, 220), (300, 200)])
    result.save("output.png")
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import splprep, splev


def add_arrow(
    image: Union[str, np.ndarray, Image.Image],
    points: Sequence[Tuple[float, float]],
    color: Union[str, Tuple[int, ...]] = "red",
    line_width: int = 3,
    arrow_head_length: float = 20.0,
    arrow_head_angle: float = 30.0,
    num_interpolated: int = 300,
    smoothing: float = 0.0,
) -> Image.Image:
    """Draw a smooth arrow on *image* that passes through *points*.

    Parameters
    ----------
    image : str | np.ndarray | PIL.Image.Image
        The background image. A file path, numpy array (H×W×C, uint8), or
        PIL Image.
    points : sequence of (x, y) tuples
        At least 2 control points that define the arrow path.  The arrow is
        drawn from the first point towards the last point.
    color : str or tuple
        Arrow colour (any format accepted by PIL).
    line_width : int
        Width of the arrow shaft in pixels.
    arrow_head_length : float
        Length of each side of the arrowhead in pixels.
    arrow_head_angle : float
        Half-angle of the arrowhead opening, in degrees.
    num_interpolated : int
        Number of sample points along the fitted spline (higher = smoother).
    smoothing : float
        Spline smoothing factor passed to ``scipy.interpolate.splprep``.
        0 means the spline passes exactly through the given points.

    Returns
    -------
    PIL.Image.Image
        A copy of the input image with the arrow drawn on top.
    """
    # --- Load / normalise the image ----------------------------------------
    if isinstance(image, str):
        img = Image.open(image).convert("RGBA")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGBA")
    elif isinstance(image, Image.Image):
        img = image.copy().convert("RGBA")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (N, 2) array of (x, y) coordinates")
    if len(pts) < 2:
        raise ValueError("Need at least 2 points to define an arrow")

    # --- Fit a parametric spline through the points ------------------------
    x, y = pts[:, 0], pts[:, 1]

    if len(pts) == 2:
        # With only 2 points, use simple linear interpolation
        t_fine = np.linspace(0, 1, num_interpolated)
        xs = x[0] + (x[1] - x[0]) * t_fine
        ys = y[0] + (y[1] - y[0]) * t_fine
    else:
        # Cubic (or lower-degree) B-spline
        k = min(3, len(pts) - 1)
        tck, _ = splprep([x, y], s=smoothing, k=k)
        t_fine = np.linspace(0, 1, num_interpolated)
        xs, ys = splev(t_fine, tck)

    # --- Draw the arrow shaft on a transparent overlay ---------------------
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    curve_coords = list(zip(xs.tolist(), ys.tolist()))
    # Draw connected line segments along the spline
    draw.line(curve_coords, fill=color, width=line_width, joint="curve")

    # --- Compute arrowhead at the tip (last point) -------------------------
    # Direction: use the last small segment of the spline for the tangent
    tip_x, tip_y = xs[-1], ys[-1]
    dx = xs[-1] - xs[-2]
    dy = ys[-1] - ys[-2]
    angle = math.atan2(dy, dx)

    ha = math.radians(arrow_head_angle)
    # Two barb endpoints
    x1 = tip_x - arrow_head_length * math.cos(angle - ha)
    y1 = tip_y - arrow_head_length * math.sin(angle - ha)
    x2 = tip_x - arrow_head_length * math.cos(angle + ha)
    y2 = tip_y - arrow_head_length * math.sin(angle + ha)

    draw.polygon([(tip_x, tip_y), (x1, y1), (x2, y2)], fill=color)

    # --- Composite and return ----------------------------------------------
    result = Image.alpha_composite(img, overlay)
    return result


# ---------------------------------------------------------------------------
# Quick demo / CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Draw an arrow on an image.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "points",
        help='Comma-separated points, e.g. "100,200;150,180;300,200"',
    )
    parser.add_argument("-o", "--output", default="output.png", help="Output path")
    parser.add_argument("--color", default="red", help="Arrow colour")
    parser.add_argument("--width", type=int, default=3, help="Line width")
    parser.add_argument("--head-length", type=float, default=20, help="Arrowhead size")
    args = parser.parse_args()

    pts = [tuple(map(float, p.split(","))) for p in args.points.split(";")]

    result = add_arrow(
        args.image,
        pts,
        color=args.color,
        line_width=args.width,
        arrow_head_length=args.head_length,
    )
    result.save(args.output)
    print(f"Saved to {args.output}")
