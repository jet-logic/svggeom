from typing import Sequence
from ...geom.bezier import Bezier
from ...geom.vector import vec
from ..pathlc import PathLC
from ..pathlc.command import BaseLC, CubicLC, MoveLC


def pathlc_to_beziers_simple(path_lc: PathLC) -> list[Bezier]:
    """
    Simple conversion of PathLC to Bezier list by extracting cubic segments.
    Converts all segments to cubic bezier representation.
    """
    beziers: list[Bezier] = []
    current_bezier = None

    # Convert the entire path to cubic representation first
    cubic_path = path_lc._tail.as_curve() if path_lc._tail else None

    if not cubic_path:
        return beziers

    # Collect all cubic segments
    segments = []
    cur: BaseLC | None = cubic_path
    while cur:
        segments.append(cur)
        cur = cur._prev

    # Process in forward order
    for segment in reversed(segments):
        if isinstance(segment, MoveLC):
            # Start new bezier
            if current_bezier and len(current_bezier) > 0:
                beziers.append(current_bezier)
            current_bezier = Bezier()
            current_bezier.move_to(segment.to())

        elif isinstance(segment, CubicLC):
            if current_bezier:
                current_bezier.curve_to(segment.c1, segment.c2, segment.to())

    # Add the final bezier
    if current_bezier and len(current_bezier) > 0:
        beziers.append(current_bezier)

    return beziers


def beziers_to_pathlc_advanced(beziers: list["Bezier"]) -> "PathLC":
    if not beziers:
        return PathLC(BaseLC.parse("M0,0"))

    path_tail: BaseLC | None = None

    for i, bezier in enumerate(beziers):
        if not bezier:
            continue

        # Start new subpath
        first_point = bezier[0][0]
        if path_tail is None:
            path_tail = MoveLC(None, first_point)
        else:
            path_tail = path_tail.move_to(first_point)

        # Process segments
        for j in range(1, len(bezier)):
            prev_pt = bezier[j - 1]
            curr_pt = bezier[j]
            v_prev, i_prev, o_prev = prev_pt
            v_curr, i_curr, o_curr = curr_pt

            # Recover absolute control points from tangents
            c1_abs = (v_prev[0] + o_prev[0], v_prev[1] + o_prev[1])
            c2_abs = (v_curr[0] + i_curr[0], v_curr[1] + i_curr[1])

            # Check if this is a straight line: both tangents are zero
            is_straight = (
                abs(o_prev[0]) < 1e-12
                and abs(o_prev[1]) < 1e-12
                and abs(i_curr[0]) < 1e-12
                and abs(i_curr[1]) < 1e-12
            )

            if is_straight:
                path_tail = path_tail.line_to(v_curr)
            else:
                path_tail = path_tail.curve_to(c1_abs, c2_abs, v_curr)

        # Close path if geometrically closed
        if bezier.is_closed():
            path_tail = path_tail.close()

    assert path_tail is not None
    return PathLC(path_tail)


def beziers_to_pathlc(beziers: list[Bezier]) -> PathLC:
    """
    Convert a list of Bezier objects to a PathLC instance.

    Args:
        beziers: List of Bezier objects to convert

    Returns:
        PathLC instance representing the bezier curves
    """
    if not beziers:
        return PathLC(BaseLC.parse("M0,0"))

    path_tail: None | BaseLC = None

    for i, bezier in enumerate(beziers):
        n = len(bezier)
        if n <= 0:
            continue

        # Start with move command for first point of each bezier
        if i == 0:
            # First bezier - start with move to first point
            first_point = bezier[0][0]
            path_tail = MoveLC(None, first_point)
        else:
            assert path_tail
            # Subsequent beziers - move to first point (start new subpath)
            first_point = bezier[0][0]
            path_tail = path_tail.move_to(first_point)

        # Process each segment in the bezier
        for j in range(1, n):
            prev_point_data = bezier[j - 1]
            curr_point_data = bezier[j]

            prev_vertex, prev_in_tangent, prev_out_tangent = prev_point_data
            curr_vertex, curr_in_tangent, curr_out_tangent = curr_point_data
            # is last
            if (j + 1) == n:
                # is line
                if all(x == 0 for x in (*prev_out_tangent, *curr_in_tangent)):
                    # is same from first vertex
                    if curr_vertex == first_point:
                        path_tail = path_tail.close()
                        continue

            # Convert to absolute control points
            c1_abs = (
                prev_vertex[0] - prev_out_tangent[0],
                prev_vertex[1] - prev_out_tangent[1],
            )
            c2_abs = (
                curr_vertex[0] - curr_in_tangent[0],
                curr_vertex[1] - curr_in_tangent[1],
            )

            # Add cubic curve segment
            path_tail = path_tail.curve_to(c1_abs, c2_abs, curr_vertex)

    assert path_tail

    return PathLC(path_tail)


####################


def interpolate_beziers_linear(
    bezier1: Bezier, bezier2: Bezier, t: float, **kwargs
) -> Bezier:
    """
    Linear interpolation between two Bezier curves.

    Args:
        bezier1: First bezier curve
        bezier2: Second bezier curve
        t: Interpolation parameter (0.0 = bezier1, 1.0 = bezier2)

    Returns:
        Interpolated bezier curve
    """
    interpolated = Bezier()
    t = max(0.0, min(1.0, t))  # Clamp to [0, 1]

    # Handle different point counts by using the maximum
    max_points = max(len(bezier1), len(bezier2))
    if kwargs.get("add_even_intermediate_points"):
        if (d := (max_points - len(bezier1))) > 0:
            bezier1 = add_even_intermediate_points(bezier1, d)
        elif (d := (max_points - len(bezier2))) > 0:
            bezier2 = add_even_intermediate_points(bezier2, d)

    if len(bezier1) > len(bezier2):
        for p2, p1_list in distribute_proportional(bezier2, bezier1):
            for i, p1 in enumerate(p1_list):
                # p1 -> p2
                p1_v, p1_i, p1_o = p1
                p2_v, p2_i, p2_o = p2
                # if i > 0:
                #     p2_i = p2_o = (0, 0)
                interpolated.add_point(
                    v=vec(p1_v).lerp(p2_v, t),
                    i=vec(p1_i).lerp(p2_i, t),
                    o=vec(p1_o).lerp(p2_o, t),
                )

    elif len(bezier1) < len(bezier2):
        for p1, p2_list in distribute_proportional(bezier1, bezier2):
            for i, p2 in enumerate(p2_list):
                # p1 -> p2
                p1_v, p1_i, p1_o = p1
                p2_v, p2_i, p2_o = p2
                # if i > 0:
                #     p2_i = p2_o = (0, 0)
                interpolated.add_point(
                    v=vec(p1_v).lerp(p2_v, t),
                    i=vec(p1_i).lerp(p2_i, t),
                    o=vec(p1_o).lerp(p2_o, t),
                )
    else:
        for j in range(max_points):
            p1_v, p1_i, p1_o = bezier1[j]
            p2_v, p2_i, p2_o = bezier2[j]
            interpolated.add_point(
                v=vec(p1_v).lerp(p2_v, t),
                i=vec(p1_i).lerp(p2_i, t),
                o=vec(p1_o).lerp(p2_o, t),
            )

    return interpolated


def interpolate_bezier_lists_linear(
    list1: list[Bezier], list2: list[Bezier], t: float, **kwargs
) -> list[Bezier]:
    """
    Linear interpolation between two lists of Bezier curves.

    Args:
        list1: First list of bezier curves
        list2: Second list of bezier curves
        t: Interpolation parameter (0.0 = list1, 1.0 = list2)

    Returns:
        Interpolated list of bezier curves
    """
    t = max(0.0, min(1.0, t))

    # Handle different list lengths by using the maximum
    max_length = max(len(list1), len(list2))
    interpolated_list = []

    if len(list1) > len(list2):
        for b2, b1_list in distribute_proportional(list2, list1):
            for b1 in b1_list:
                # b1 -> b2
                interpolated_bezier = interpolate_beziers_linear(b1, b2, t, **kwargs)
                interpolated_list.append(interpolated_bezier)
    elif len(list1) < len(list2):
        for b1, b2_list in distribute_proportional(list1, list2):
            for b2 in b2_list:
                # b1 -> b2
                interpolated_bezier = interpolate_beziers_linear(b1, b2, t, **kwargs)
                interpolated_list.append(interpolated_bezier)
    else:
        for i in range(max_length):
            interpolated_bezier = interpolate_beziers_linear(
                list1[i], list2[i], t, **kwargs
            )
            interpolated_list.append(interpolated_bezier)
    return interpolated_list


def bezier_list_center_bbox(beziers: list[Bezier]) -> tuple[float, float]:
    """
    Compute center as the center of the combined bounding box of all curves.
    Fast and good for overall shape containment.
    """
    if not beziers:
        return (0.0, 0.0)

    combined_bbox = None

    for bezier in beziers:
        bbox = bezier.bounding_box()
        if bbox:
            if combined_bbox is None:
                combined_bbox = bbox
            else:
                combined_bbox += bbox

    if combined_bbox is None:
        return (0.0, 0.0)

    return (combined_bbox.center_x, combined_bbox.center_y)


def split_longest(b: Bezier):
    pts = b
    lens: list[tuple[float, int]] = []
    for j, (v_cur, i, o) in enumerate(pts):
        if j > 0:
            v_prev = pts[j - 1][0]
            lens.append(((vec(v_prev) - vec(v_cur)).abs(), j))
    lens.sort()
    j = lens[-1][1]
    v1 = b[j - 1]
    v2 = b[j]
    p1 = vec(v1[0])
    p2 = vec(v2[0])
    c1 = p1 + v1[2]
    c2 = p2 + v2[1]
    cA, cB = MoveLC(None, p1).curve_to(c1, c2, p2).split_at(0.5)
    assert isinstance(cA, CubicLC)
    assert isinstance(cB, CubicLC)
    v = cA.to()
    i = cA.c2 - v
    o = cB.c1 - v
    b.insert_point(j, v, i, o)


#     # b.insert_point


def distribute_proportional(list1: Sequence[object], list2: Sequence[object]):
    """Distribute list2 elements proportionally based on list1 length"""
    assert len(list1) > 0 and len(list2) > len(list1)
    result: list[tuple[object, Sequence[object]]] = []

    # Calculate how many elements each item in list1 should get
    base_chunk = len(list2) // len(list1)
    remainder = len(list2) % len(list1)

    start = 0
    for i, item in enumerate(list1):
        # Items get base_chunk + 1 if there are remainders to distribute
        chunk_size = base_chunk + (1 if i < remainder else 0)
        end = start + chunk_size

        result.append((item, list2[start:end]))
        start = end

    return result


def d_interpolator(d1="", d2="", **kwargs):
    p1 = PathLC.parse(d1)
    p2 = PathLC.parse(d2)
    b1 = pathlc_to_beziers_simple(p1)
    b2 = pathlc_to_beziers_simple(p2)
    reverse = kwargs.get("reverse")
    if reverse == 2:
        b2 = list(map(lambda b: b.reverse_dir(), b2))
    elif reverse:
        b1 = list(map(lambda b: b.reverse_dir(), b1))

    def fn(t=0):
        bt = interpolate_bezier_lists_linear(b1, b2, t, **kwargs)
        return beziers_to_pathlc(bt).describe()

    return fn


from ..bezier import Bezier
from ..help.cubichelp import cubic_point_at, cubic_length
from ..vector import vec
from typing import List
import bisect


def cubic_segment_length(points: Sequence[tuple[float, float]]) -> float:
    """Estimate length of a cubic segment using recursive flattening."""
    return cubic_length(points)


def sample_curve_evenly(bezier: Bezier, total_samples: int) -> List[float]:
    """
    Return a list of `t` values in [0, 1] that correspond to approximately
    equally spaced points along the full Bezier curve.
    """
    if len(bezier) < 2 or total_samples <= 0:
        return []

    # Reconstruct cubic segments
    segments = []
    seg_starts = [0.0]
    total_len = 0.0

    for i in range(1, len(bezier)):
        v0, i0, o0 = bezier[i - 1]
        v1, i1, o1 = bezier[i]

        c1 = (v0[0] + o0[0], v0[1] + o0[1])
        c2 = (v1[0] + i1[0], v1[1] + i1[1])
        seg: list[tuple[float, float]] = [v0, c1, c2, v1]
        segments.append(seg)
        seg_len = cubic_segment_length(seg)
        total_len += seg_len
        seg_starts.append(total_len)

    if total_len == 0:
        return []

    # Generate equally spaced distances
    step = total_len / (total_samples + 1)
    t_vals = []

    for k in range(1, total_samples + 1):
        target = k * step

        # Find which segment contains this arc length
        idx = bisect.bisect_right(seg_starts, target) - 1
        idx = max(0, min(idx, len(segments) - 1))

        seg = segments[idx]
        seg_start = seg_starts[idx]
        seg_len = seg_starts[idx + 1] - seg_start

        if seg_len <= 0:
            local_t = 0.0
        else:
            local_t = (target - seg_start) / seg_len
            local_t = max(0.0, min(1.0, local_t))

        # Map local t to global t: each segment is 1 unit in the abstract parameter space
        global_t = (idx + local_t) / len(segments)
        t_vals.append(global_t)

    return sorted(set(t_vals))


def add_even_intermediate_points(bezier: Bezier, num_points: int) -> Bezier:
    """
    Insert `num_points` new vertices **between** existing vertices of the Bezier,
    spaced approximately evenly by arc length, while **preserving all original vertices**.

    The shape remains **exactly identical** to the original.
    """
    if num_points <= 0:
        return Bezier(bezier)  # shallow copy

    # Get global t values where to sample (excluding t=0 and t=1)
    t_vals = sample_curve_evenly(bezier, num_points)
    # Filter out endpoints to avoid duplicating original vertices
    t_vals = [t for t in t_vals if 0.0 < t < 1.0]

    if not t_vals:
        return Bezier(bezier)

    # Build list of cubic segments with metadata
    segments = []  # each: (v0, o0, v1, i1, segment_index)
    for i in range(1, len(bezier)):
        v0, i0, o0 = bezier[i - 1]
        v1, i1, o1 = bezier[i]
        segments.append((v0, o0, v1, i1, i - 1))

    # Create result
    result = Bezier()
    for pt in bezier:
        result.add_point(*pt)

    # Insert points in reverse order to preserve indices
    for t in reversed(t_vals):
        # Determine which segment this t falls into
        seg_idx = min(int(t * len(segments)), len(segments) - 1)
        local_t = (t * len(segments)) - seg_idx
        local_t = max(0.0, min(1.0, local_t))

        v0, o0, v1, i1, orig_idx = segments[seg_idx]

        # Reconstruct cubic control points
        c1 = (v0[0] + o0[0], v0[1] + o0[1])
        c2 = (v1[0] + i1[0], v1[1] + i1[1])
        cubic = [v0, c1, c2, v1]

        # Evaluate point and nearby points for tangents
        pt = vec(cubic_point_at(cubic, local_t))
        eps = 1e-6
        pt_back = vec(cubic_point_at(cubic, max(0.0, local_t - eps)))
        pt_fwd = vec(cubic_point_at(cubic, min(1.0, local_t + eps)))

        in_tan = pt_back - pt
        out_tan = pt_fwd - pt

        # Insert after original vertex at `orig_idx`
        result.insert_point(orig_idx + 1, pt, in_tan, out_tan)

        # Update segments after insertion (shift indices)
        for j in range(len(segments)):
            if segments[j][4] >= orig_idx + 1:
                v0, o0, v1, i1, idx = segments[j]
                segments[j] = (v0, o0, v1, i1, idx + 1)

    return result
