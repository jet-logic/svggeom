"""
Shape morphing between SVG paths.

Handles:
  - different vertex counts        (arc-length resampling)
  - different numbers of shapes    (contour matching + degenerate pairs)
  - holes                          (containment classification + winding normalize)
  - line, cubic, quad, smooth, arc commands
  - absolute & relative, M->L implicit lineto

Usage:
    interp = SVGPathInterpolator(src_d, tgt_d)
    d_mid  = interp.interpolate(0.5)
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

Point = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Path parsing
# --------------------------------------------------------------------------- #

_CMD_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def _tokenize(d: str) -> List[Tuple[str, List[float]]]:
    """Split an SVG path `d` into (command_letter, [floats]) tuples."""
    out = []
    i, n = 0, len(d)
    while i < n:
        c = d[i]
        if c.isspace() or c == ",":
            i += 1
            continue
        if not _CMD_RE.match(c):
            i += 1
            continue
        cmd = c
        i += 1
        args: List[float] = []
        while i < n:
            while i < n and (d[i].isspace() or d[i] == ","):
                i += 1
            if i >= n or _CMD_RE.match(d[i]):
                break
            m = _NUM_RE.match(d, i)
            if not m:
                break
            args.append(float(m.group()))
            i = m.end()
        out.append((cmd, args))
    return out


def _angle(ux, uy, vx, vy) -> float:
    return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)


def _arc_points(x1, y1, rx, ry, rot_deg, large, sweep, x2, y2, steps):
    """Sample an SVG elliptical arc into `steps` points (endpoint param)."""
    if rx == 0 or ry == 0:
        return [(x2, y2)]
    rx, ry = abs(rx), abs(ry)
    phi = math.radians(rot_deg)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cp * dx + sp * dy
    y1p = -sp * dx + cp * dy

    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1:
        s = math.sqrt(lam)
        rx *= s
        ry *= s

    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    co = 0.0 if den == 0 else math.sqrt(max(0.0, num / den))
    if large == sweep:
        co = -co

    cxp = co * rx * y1p / ry
    cyp = -co * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2.0
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2.0

    theta1 = _angle(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = _angle(
        (x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry
    )
    if not sweep and dtheta > 0:
        dtheta -= 2 * math.pi
    elif sweep and dtheta < 0:
        dtheta += 2 * math.pi

    pts = []
    for k in range(1, steps + 1):
        t = theta1 + dtheta * k / steps
        ct, st = math.cos(t), math.sin(t)
        pts.append((cx + rx * ct * cp - ry * st * sp, cy + rx * ct * sp + ry * st * cp))
    return pts


# --------------------------------------------------------------------------- #
# Contour representation
# --------------------------------------------------------------------------- #


@dataclass
class Contour:
    points: List[Point]
    closed: bool = True
    is_hole: bool = False

    def signed_area(self) -> float:
        n = len(self.points)
        if n < 3:
            return 0.0
        s = 0.0
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return s / 2.0

    def centroid(self) -> Point:
        n = len(self.points)
        if n == 0:
            return (0.0, 0.0)
        a = self.signed_area()
        if abs(a) < 1e-12:
            return (
                sum(p[0] for p in self.points) / n,
                sum(p[1] for p in self.points) / n,
            )
        cx = cy = 0.0
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            cr = x1 * y2 - x2 * y1
            cx += (x1 + x2) * cr
            cy += (y1 + y2) * cr
        return (cx / (6.0 * a), cy / (6.0 * a))


def _point_in_polygon(pt: Point, poly: Sequence[Point]) -> bool:
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def _interior_point(poly: Sequence[Point]) -> Point:
    """Robustly find a point strictly inside a simple polygon."""
    n = len(poly)
    if n < 3:
        return poly[0] if poly else (0.0, 0.0)
    ys = [p[1] for p in poly]
    y_min, y_max = min(ys), max(ys)
    for frac in (0.5, 0.37, 0.63, 0.21, 0.79, 0.13, 0.87):
        y_test = y_min + (y_max - y_min) * frac
        xs = []
        j = n - 1
        for i in range(n):
            yi, yj = poly[i][1], poly[j][1]
            if (yi > y_test) != (yj > y_test):
                t = (y_test - yi) / (yj - yi)
                xs.append(poly[i][0] + t * (poly[j][0] - poly[i][0]))
            j = i
        if len(xs) >= 2:
            xs.sort()
            for k in range(0, len(xs) - 1, 2):
                if xs[k + 1] - xs[k] > 1e-9:
                    return ((xs[k] + xs[k + 1]) / 2.0, y_test)
    return Contour(list(poly)).centroid()


# --------------------------------------------------------------------------- #
# Path  ->  contours
# --------------------------------------------------------------------------- #


def path_to_contours(d: str, curve_steps: int = 16) -> List[Contour]:
    cmds = _tokenize(d)
    contours: List[Contour] = []
    cur: List[Point] = []
    cx = cy = sx = sy = 0.0
    prev_cmd = ""
    prev_q: Optional[Point] = None
    prev_c: Optional[Point] = None

    def push(p):
        cur.append(p)

    for cmd, args in cmds:
        up = cmd.upper()
        rel = cmd.islower()

        if up == "Z":
            if cur:
                contours.append(Contour(cur, True))
                cur = []
            cx, cy = sx, sy
            prev_cmd, prev_q, prev_c = "Z", None, None
            continue

        if up == "M":
            if cur:
                contours.append(Contour(cur, True))
                cur = []
            x, y = args[0], args[1]
            if rel:
                x += cx
                y += cy
            cx, cy = x, y
            sx, sy = x, y
            cur = [(cx, cy)]
            i = 2
            while i + 1 < len(args):
                x, y = args[i], args[i + 1]
                if rel:
                    x += cx
                    y += cy
                cx, cy = x, y
                push((cx, cy))
                i += 2
            prev_cmd = "L" if len(args) > 2 else "M"
            prev_q = prev_c = None

        elif up == "L":
            i = 0
            while i + 1 < len(args):
                x, y = args[i], args[i + 1]
                if rel:
                    x += cx
                    y += cy
                cx, cy = x, y
                push((cx, cy))
                i += 2
            prev_cmd, prev_q, prev_c = "L", None, None

        elif up == "H":
            for v in args:
                cx = v + cx if rel else v
                push((cx, cy))
            prev_cmd, prev_q, prev_c = "H", None, None

        elif up == "V":
            for v in args:
                cy = v + cy if rel else v
                push((cx, cy))
            prev_cmd, prev_q, prev_c = "V", None, None

        elif up == "C":
            i = 0
            while i + 5 < len(args):
                x1, y1, x2, y2, x, y = args[i : i + 6]
                if rel:
                    x1 += cx
                    y1 += cy
                    x2 += cx
                    y2 += cy
                    x += cx
                    y += cy
                for k in range(1, curve_steps + 1):
                    t = k / curve_steps
                    mt = 1 - t
                    px = (
                        mt**3 * cx
                        + 3 * mt * mt * t * x1
                        + 3 * mt * t * t * x2
                        + t**3 * x
                    )
                    py = (
                        mt**3 * cy
                        + 3 * mt * mt * t * y1
                        + 3 * mt * t * t * y2
                        + t**3 * y
                    )
                    push((px, py))
                prev_c = (x2, y2)
                cx, cy = x, y
                i += 6
            prev_cmd, prev_q = "C", None

        elif up == "S":
            i = 0
            while i + 3 < len(args):
                x2, y2, x, y = args[i : i + 4]
                if rel:
                    x2 += cx
                    y2 += cy
                    x += cx
                    y += cy
                if prev_cmd in ("C", "S") and prev_c is not None:
                    x1, y1 = 2 * cx - prev_c[0], 2 * cy - prev_c[1]
                else:
                    x1, y1 = cx, cy
                for k in range(1, curve_steps + 1):
                    t = k / curve_steps
                    mt = 1 - t
                    px = (
                        mt**3 * cx
                        + 3 * mt * mt * t * x1
                        + 3 * mt * t * t * x2
                        + t**3 * x
                    )
                    py = (
                        mt**3 * cy
                        + 3 * mt * mt * t * y1
                        + 3 * mt * t * t * y2
                        + t**3 * y
                    )
                    push((px, py))
                prev_c = (x2, y2)
                cx, cy = x, y
                i += 4
            prev_cmd, prev_q = "S", None

        elif up == "Q":
            i = 0
            while i + 3 < len(args):
                x1, y1, x, y = args[i : i + 4]
                if rel:
                    x1 += cx
                    y1 += cy
                    x += cx
                    y += cy
                for k in range(1, curve_steps + 1):
                    t = k / curve_steps
                    mt = 1 - t
                    push(
                        (
                            mt * mt * cx + 2 * mt * t * x1 + t * t * x,
                            mt * mt * cy + 2 * mt * t * y1 + t * t * y,
                        )
                    )
                prev_q = (x1, y1)
                cx, cy = x, y
                i += 4
            prev_cmd, prev_c = "Q", None

        elif up == "T":
            i = 0
            while i + 1 < len(args):
                x, y = args[i], args[i + 1]
                if rel:
                    x += cx
                    y += cy
                if prev_cmd in ("Q", "T") and prev_q is not None:
                    x1, y1 = 2 * cx - prev_q[0], 2 * cy - prev_q[1]
                else:
                    x1, y1 = cx, cy
                for k in range(1, curve_steps + 1):
                    t = k / curve_steps
                    mt = 1 - t
                    push(
                        (
                            mt * mt * cx + 2 * mt * t * x1 + t * t * x,
                            mt * mt * cy + 2 * mt * t * y1 + t * t * y,
                        )
                    )
                prev_q = (x1, y1)
                cx, cy = x, y
                i += 2
            prev_cmd, prev_c = "T", None

        elif up == "A":
            i = 0
            while i + 6 < len(args):
                rx, ry, rot, la, sw, x, y = args[i : i + 7]
                if rel:
                    x += cx
                    y += cy
                for p in _arc_points(
                    cx, cy, rx, ry, rot, int(la) != 0, int(sw) != 0, x, y, curve_steps
                ):
                    push(p)
                cx, cy = x, y
                i += 7
            prev_cmd, prev_q, prev_c = "A", None, None

    if cur:
        contours.append(Contour(cur, True))
    return contours


# --------------------------------------------------------------------------- #
# Contour classification & winding normalisation
# --------------------------------------------------------------------------- #


def classify_contours(contours: List[Contour]) -> None:
    """Mark each contour as hole/outer using ray-cast containment."""
    for i, c in enumerate(contours):
        if len(c.points) < 3:
            c.is_hole = False
            continue
        rep = _interior_point(c.points)
        depth = 0
        for j, other in enumerate(contours):
            if i == j or len(other.points) < 3:
                continue
            if _point_in_polygon(rep, other.points):
                depth += 1
        c.is_hole = depth % 2 == 1


def normalize_winding(contours: List[Contour]) -> None:
    """Outers one direction, holes the other -> so morph frames stay consistent."""
    for c in contours:
        a = c.signed_area()
        if c.is_hole and a > 0:
            c.points = c.points[::-1]
        elif not c.is_hole and a < 0:
            c.points = c.points[::-1]


# --------------------------------------------------------------------------- #
# Resampling & alignment
# --------------------------------------------------------------------------- #


def resample(contour: Contour, n: int) -> List[Point]:
    """Return `n` points uniformly spaced along the contour by arc length."""
    pts = contour.points
    if not pts:
        return [(0.0, 0.0)] * n
    if len(pts) == 1:
        return [pts[0]] * n

    ext = pts + [pts[0]] if contour.closed else list(pts)
    cum = [0.0]
    for i in range(1, len(ext)):
        cum.append(
            cum[-1] + math.hypot(ext[i][0] - ext[i - 1][0], ext[i][1] - ext[i - 1][1])
        )
    total = cum[-1]
    if total < 1e-12:
        return [pts[0]] * n

    if contour.closed:
        ts = [i * total / n for i in range(n)]
    else:
        ts = [i * total / (n - 1) for i in range(n)] if n > 1 else [0.0]

    out, j = [], 0
    for t in ts:
        while j < len(cum) - 2 and cum[j + 1] < t:
            j += 1
        seg = cum[j + 1] - cum[j]
        f = (t - cum[j]) / seg if seg > 1e-12 else 0.0
        f = min(max(f, 0.0), 1.0)
        out.append(
            (
                ext[j][0] + f * (ext[j + 1][0] - ext[j][0]),
                ext[j][1] + f * (ext[j + 1][1] - ext[j][1]),
            )
        )
    return out


def align_contours(src: List[Point], tgt: List[Point]) -> List[Point]:
    """Rotate `tgt` so its vertices best line up with `src` (closed shapes)."""
    n = len(src)
    if n == 0:
        return tgt

    def cost(k):
        c = 0.0
        for i in range(n):
            j = (i + k) % n
            dx = src[i][0] - tgt[j][0]
            dy = src[i][1] - tgt[j][1]
            c += dx * dx + dy * dy
        return c

    step = max(1, n // 32)
    best_k, best_c = 0, float("inf")
    for k in range(0, n, step):
        c = cost(k)
        if c < best_c:
            best_c, best_k = c, k
    for k in range(max(0, best_k - step), min(n, best_k + step + 1)):
        c = cost(k)
        if c < best_c:
            best_c, best_k = c, k
    return [tgt[(i + best_k) % n] for i in range(n)]


# --------------------------------------------------------------------------- #
# Contour matching
# --------------------------------------------------------------------------- #


def _match_cost(a: Contour, b: Contour, bbox_diag: float) -> float:
    aa, bb = abs(a.signed_area()), abs(b.signed_area())
    area_cost = 0.0 if max(aa, bb) < 1e-12 else abs(aa - bb) / max(aa, bb)
    ca, cb = a.centroid(), b.centroid()
    dist_cost = math.hypot(ca[0] - cb[0], ca[1] - cb[1]) / max(bbox_diag, 1e-12)
    return area_cost + dist_cost


def match_contours(src: List[Contour], tgt: List[Contour], bbox_diag: float):
    """Greedy pairing (holes<->holes, outers<->outers).  Leftovers pair with None."""
    src_o = [c for c in src if not c.is_hole]
    src_h = [c for c in src if c.is_hole]
    tgt_o = [c for c in tgt if not c.is_hole]
    tgt_h = [c for c in tgt if c.is_hole]

    pairs: List[Tuple[Optional[Contour], Optional[Contour]]] = []
    for s_list, t_list in ((src_o, tgt_o), (src_h, tgt_h)):
        cand = sorted(
            (_match_cost(s, t, bbox_diag), i, j)
            for i, s in enumerate(s_list)
            for j, t in enumerate(t_list)
        )
        used_s, used_t = set(), set()
        for _, i, j in cand:
            if i in used_s or j in used_t:
                continue
            used_s.add(i)
            used_t.add(j)
            pairs.append((s_list[i], t_list[j]))
        for i, s in enumerate(s_list):
            if i not in used_s:
                pairs.append((s, None))
        for j, t in enumerate(t_list):
            if j not in used_t:
                pairs.append((None, t))
    return pairs


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def contours_to_svg_path(contours: List[Contour], precision: int = 3) -> str:
    parts = []
    for c in contours:
        if not c.points:
            continue
        x, y = c.points[0]
        parts.append(f"M {x:.{precision}f} {y:.{precision}f}")
        for p in c.points[1:]:
            parts.append(f"L {p[0]:.{precision}f} {p[1]:.{precision}f}")
        if c.closed:
            parts.append("Z")
    return " ".join(parts)


def svg_document(
    d: str, width: float, height: float, fill: str = "black", fill_rule: str = "evenodd"
) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<path d="{d}" fill="{fill}" fill-rule="{fill_rule}"/>'
        f"</svg>"
    )


# --------------------------------------------------------------------------- #
# Main API
# --------------------------------------------------------------------------- #


class SVGPathInterpolator:
    """
    Pre-computes vertex correspondences between two SVG path strings.
    Call `.interpolate(t)` for t in [0, 1] to get a new `d` string.

    Parameters
    ----------
    src_d, tgt_d            : SVG path data
    samples_per_contour     : points per contour in the output (>= 8)
    curve_steps             : polyline samples per curve command while flattening
    close_open_contours     : treat subpaths without Z as filled (default True)
    """

    def __init__(
        self,
        src_d: str,
        tgt_d: str,
        samples_per_contour: int = 128,
        curve_steps: int = 16,
        close_open_contours: bool = True,
    ):
        self.n = max(8, samples_per_contour)

        src = path_to_contours(src_d, curve_steps)
        tgt = path_to_contours(tgt_d, curve_steps)
        if close_open_contours:
            for c in src:
                c.closed = True
            for c in tgt:
                c.closed = True

        classify_contours(src)
        classify_contours(tgt)
        normalize_winding(src)
        normalize_winding(tgt)

        all_pts = [p for c in src + tgt for p in c.points]
        if all_pts:
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            bbox_diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
        else:
            bbox_diag = 1.0

        # pre-compute every pair's resampled/aligned points
        self._pairs: List[dict] = []
        for s, t in match_contours(src, tgt, bbox_diag):
            if s is not None and t is not None:
                sp = resample(s, self.n)
                tp = align_contours(sp, resample(t, self.n))
                self._pairs.append({"src": sp, "tgt": tp, "closed": s.closed})
            elif s is not None:
                sp = resample(s, self.n)
                cen = s.centroid()
                self._pairs.append(
                    {"src": sp, "tgt": [cen] * self.n, "closed": s.closed}
                )
            else:
                tp = resample(t, self.n)
                cen = t.centroid()
                self._pairs.append(
                    {"src": [cen] * self.n, "tgt": tp, "closed": t.closed}
                )

    # ---- public ---------------------------------------------------------- #

    def interpolate_contours(self, t: float) -> List[Contour]:
        t = max(0.0, min(1.0, t))
        out = []
        for pd in self._pairs:
            src, tgt = pd["src"], pd["tgt"]
            pts = [
                ((1 - t) * s[0] + t * g[0], (1 - t) * s[1] + t * g[1])
                for s, g in zip(src, tgt)
            ]
            out.append(Contour(pts, pd["closed"]))
        return out

    def interpolate(self, t: float) -> str:
        return contours_to_svg_path(self.interpolate_contours(t))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import os

    # Square-with-square-hole  ->  Circle-with-circular-hole
    src = "M 10 10 L 90 10 L 90 90 L 10 90 Z " "M 35 35 L 35 65 L 65 65 L 65 35 Z"
    tgt = (
        "M 10 50 A 40 40 0 1 0 90 50 A 40 40 0 1 0 10 50 Z "
        "M 35 50 A 15 15 0 1 0 65 50 A 15 15 0 1 0 35 50 Z"
    )

    interp = SVGPathInterpolator(src, tgt, samples_per_contour=128)

    os.makedirs("frames", exist_ok=True)
    N = 20
    for i in range(N + 1):
        t = i / N
        d = interp.interpolate(t)
        svg = svg_document(d, 100, 100)
        with open(f"frames/frame_{i:03d}.svg", "w") as f:
            f.write(svg)

    print(f"Wrote {N + 1} SVG frames to ./frames/")
    print("src length:", len(src), "  tgt length:", len(tgt))
    print(
        "interpolated d at t=0.5 (first 120 chars):",
        interp.interpolate(0.5)[:120],
        "...",
    )
