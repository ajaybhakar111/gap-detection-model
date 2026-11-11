"""Row-spacing based scale estimation and gap-box finalisation (all in FEET).

Idea: crop rows are planted a known distance apart (row spacing, in feet).
Measuring that distance in pixels gives the image scale (feet per pixel)
without needing the drone's flight height. Boxes can then be judged in
real-world units using the row spacing and the plant spacing.
"""
import math

import numpy as np
import cv2
from PIL import ImageDraw, ImageFont


# ---------------------------------------------------------------- row detection
def vegetation_mask(rgb):
    """1.0 where a pixel looks like green plant (Excess Green + Otsu threshold)."""
    img = rgb.astype(np.float32)
    exg = 2 * img[..., 1] - img[..., 0] - img[..., 2]
    # Clip extreme values (shadows, bright objects) so they cannot squash the
    # useful range before thresholding.
    lo, hi = np.percentile(exg, [2, 98])
    if hi - lo < 1e-6:
        return np.zeros(exg.shape, np.float32)
    exg = np.clip((exg - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(exg, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(np.float32)


def _rotated_profile(mask, angle):
    """Rotate the mask, then average down the columns of its central crop.

    When the rows line up with the vertical axis the profile has sharp,
    regular peaks (one per row), so its variance is highest.
    """
    h, w = mask.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_LINEAR)
    ch, cw = int(h * 0.7), int(w * 0.7)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = rot[y0:y0 + ch, x0:x0 + cw]
    profile = crop.mean(axis=0)
    return float(profile.var()), profile


def _period_from_profile(profile, min_lag):
    """Dominant repeat distance of a 1-D profile via autocorrelation."""
    p = profile - profile.mean()
    n = len(p)
    ac = np.correlate(p, p, mode="full")[n - 1:]
    if ac[0] <= 1e-9:
        return None, 0.0
    ac = ac / ac[0]

    hi = n // 2
    peaks = [k for k in range(max(min_lag, 1), hi)
             if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0.1]
    if not peaks:
        return None, 0.0
    main = max(peaks, key=lambda k: ac[k])

    # Refine: look at the 2nd, 3rd... repeats and average lag / k.
    estimates, k = [], 1
    while k * main < n * 0.9:
        centre = k * main
        lo = max(int(centre - main * 0.25), 1)
        up = min(int(centre + main * 0.25), n - 2)
        if up <= lo:
            break
        lag = lo + int(np.argmax(ac[lo:up + 1]))
        estimates.append(lag / k)
        k += 1
    return float(np.median(estimates)), float(ac[main])


def estimate_row_spacing(rgb, max_side=640, max_tilt=25, step=2, min_spacing_px=5):
    """Estimate the distance between crop rows, in pixels of the ORIGINAL image.

    Returns a dict with spacing_px, orientation ('vertical' or 'horizontal'),
    tilt_deg and reliability (0-1, higher is better), or None if no regular
    row pattern was found.
    """
    mask = vegetation_mask(rgb)
    h, w = mask.shape
    scale = max_side / max(h, w)
    small = cv2.resize(mask, (max(8, int(round(w * scale))), max(8, int(round(h * scale)))),
                       interpolation=cv2.INTER_AREA)

    best = None
    for orientation, m in (("vertical", small),
                           ("horizontal", np.ascontiguousarray(np.rot90(small)))):
        for angle in np.arange(-max_tilt, max_tilt + 1e-6, step):
            score, prof = _rotated_profile(m, float(angle))
            if best is None or score > best[0]:
                best = (score, orientation, float(angle), prof)

    _, orientation, angle, prof = best
    spacing_small, reliability = _period_from_profile(prof, min_spacing_px)
    if spacing_small is None:
        return None
    return {"spacing_px": spacing_small / scale, "orientation": orientation,
            "tilt_deg": angle, "reliability": reliability}


def count_rows(img_w, img_h, orientation, row_px):
    """Estimated number of crop rows that fit across the image."""
    across_px = img_w if orientation == "vertical" else img_h
    return max(1, int(round(across_px / row_px)))


# ------------------------------------------------------------- box finalisation
def split_axes(box, orientation):
    """Return ((along-row start, end), (across-row start, end)) for a box."""
    x1, y1, x2, y2 = box[:4]
    if orientation == "vertical":
        return (y1, y2), (x1, x2)
    return (x1, x2), (y1, y2)


def merge_boxes(boxes, orientation, row_spacing_px, merge_dist_px):
    """Merge boxes in the same row that are closer than merge_dist_px along the row.

    Two boxes closer than a plant spacing (or less) leave no room for a plant
    between them, so they are treated as one gap. Overlapping boxes merge too.
    Each box is (x1, y1, x2, y2, conf).
    """
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a_al, a_ac = split_axes(boxes[i], orientation)
                b_al, b_ac = split_axes(boxes[j], orientation)
                same_row = abs((a_ac[0] + a_ac[1]) / 2 - (b_ac[0] + b_ac[1]) / 2) < 0.5 * row_spacing_px
                gap_along = max(a_al[0], b_al[0]) - min(a_al[1], b_al[1])  # < 0 means overlap
                if same_row and gap_along < merge_dist_px:
                    a, b = boxes[i], boxes[j]
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]),
                                max(a[2], b[2]), max(a[3], b[3]), max(a[4], b[4])]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return [tuple(b) for b in boxes]


def finalize_boxes(boxes, orientation, ft_per_px, row_spacing_ft, plant_spacing_ft,
                   min_len_factor=1.0, merge_factor=0.5, max_width_frac=0.7):
    """Apply the spacing-based rules (all distances in feet).

    1. Merge boxes in the same row separated by less than merge_factor x plant spacing.
    2. Drop boxes shorter (along the row) than min_len_factor x plant spacing.
    3. Drop boxes wider (across the row) than max_width_frac x row spacing,
       which are most likely bare inter-row soil rather than an in-row gap.

    Returns (kept, dropped, n_merged_away); kept/dropped are lists of dicts
    with length_ft, width_ft and area_ft2.
    """
    row_px = row_spacing_ft / ft_per_px
    merge_px = merge_factor * plant_spacing_ft / ft_per_px
    merged = merge_boxes(boxes, orientation, row_px, merge_px)

    kept, dropped = [], []
    for b in merged:
        along, across = split_axes(b, orientation)
        length_ft = (along[1] - along[0]) * ft_per_px
        width_ft = (across[1] - across[0]) * ft_per_px
        info = {"box": b[:4], "conf": b[4], "length_ft": length_ft,
                "width_ft": width_ft, "area_ft2": length_ft * width_ft}
        if length_ft < min_len_factor * plant_spacing_ft:
            info["reason"] = "shorter than minimum gap length"
            dropped.append(info)
        elif width_ft > max_width_frac * row_spacing_ft:
            info["reason"] = "too wide (likely inter-row soil)"
            dropped.append(info)
        else:
            kept.append(info)
    return kept, dropped, len(boxes) - len(merged)


# ------------------------------------------------------- numbering, planting, drawing
def plants_needed(length_ft, plant_spacing_ft, mode="nearest"):
    """Estimated number of plants to refill a gap: gap length / plant spacing.

    mode 'nearest' rounds to the closest whole plant, 'up' always rounds up.
    At least 1 plant per gap.
    """
    ratio = length_ft / plant_spacing_ft
    n = math.ceil(ratio - 1e-9) if mode == "up" else math.floor(ratio + 0.5)
    return max(1, int(n))


def order_gaps(gaps, orientation, row_px):
    """Sort gaps into reading order: row by row, then along each row.

    Gaps whose across-row centres lie within half a row spacing of the first
    gap in a group are treated as belonging to the same row.
    """
    def across_centre(g):
        _, ac = split_axes(g["box"], orientation)
        return (ac[0] + ac[1]) / 2

    def along_start(g):
        al, _ = split_axes(g["box"], orientation)
        return al[0]

    groups, current = [], []
    for g in sorted(gaps, key=across_centre):
        if current and across_centre(g) - across_centre(current[0]) > 0.5 * row_px:
            groups.append(current)
            current = []
        current.append(g)
    if current:
        groups.append(current)

    ordered = []
    for grp in groups:
        ordered.extend(sorted(grp, key=along_start))
    return ordered


def _get_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:            # very old Pillow
        return ImageFont.load_default()


def draw_gaps(image, kept):
    """Red boxes with a red numbered badge for every final gap. Nothing else is drawn."""
    out = image.copy()
    d = ImageDraw.Draw(out)
    lw = max(2, image.width // 400)
    font = _get_font(max(16, image.width // 40))

    for i, g in enumerate(kept, 1):
        x1, y1, x2, y2 = g["box"]
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=lw)
        label = str(i)
        l, t, r, b = d.textbbox((0, 0), label, font=font)
        tw, th, pad = r - l, b - t, 3
        by1 = max(0, y1 - th - 2 * pad)
        d.rectangle([x1, by1, x1 + tw + 2 * pad, by1 + th + 2 * pad], fill=(255, 0, 0))
        d.text((x1 + pad - l, by1 + pad - t), label, font=font, fill=(255, 255, 255))
    return out