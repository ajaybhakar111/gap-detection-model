import base64
import csv
import io
import math
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from ultralytics import YOLO

from spacing import (count_rows, draw_gaps, estimate_row_spacing, finalize_boxes,
                     order_gaps, plants_needed)

try:
    import fitz  # PyMuPDF, used to turn the first page of a PDF into an image
except ImportError:
    fitz = None

# ------------------------------------------------------------------- settings
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
CSV_PATH = BASE_DIR / "output" / "gap_gps_results.csv"   # optional fallback (reference code)

FT_TO_M = 0.3048

model = YOLO(str(MODEL_PATH))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------ GPS helpers
def load_gps_csv(path):
    """Fallback GPS list [(lat, lon), ...] from the reference CSV, if it exists."""
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["Latitude"]), float(r["Longitude"])))
            except (KeyError, ValueError):
                continue
    return rows


CSV_GPS = load_gps_csv(CSV_PATH)
print("GPS fallback records loaded:", len(CSV_GPS))


def _dms_to_deg(dms, ref):
    d, m, s = (float(x) for x in dms)
    v = d + m / 60 + s / 3600
    return -v if ref in ("S", "W") else v


def read_gps(raw, img):
    """Return (lat, lon, yaw_deg) from EXIF / DJI XMP. Any value may be None."""
    lat = lon = yaw = None
    try:
        gps = img.getexif().get_ifd(0x8825)
        if gps and 2 in gps and 4 in gps:
            lat = _dms_to_deg(gps[2], gps.get(1, "N"))
            lon = _dms_to_deg(gps[4], gps.get(3, "E"))
        if gps and 17 in gps:
            yaw = float(gps[17])
    except Exception:
        pass

    head = raw[:262144]
    for tag in (b"GimbalYawDegree", b"FlightYawDegree"):
        m = re.search(rb"drone-dji:" + tag + rb'\s*[=>]\s*"?([+-]?\d+(?:\.\d+)?)', head)
        if m:
            yaw = float(m.group(1))
            break
    return lat, lon, yaw


def pixel_to_latlon(cx, cy, w, h, m_per_px, lat0, lon0, yaw_deg):
    """Offset of a pixel from the image centre (nadir photo) -> lat/lon."""
    yaw = math.radians(yaw_deg or 0.0)
    fwd = -(cy - h / 2) * m_per_px      # image 'up'
    right = (cx - w / 2) * m_per_px     # image 'right'
    north = fwd * math.cos(yaw) - right * math.sin(yaw)
    east = fwd * math.sin(yaw) + right * math.cos(yaw)
    lat = lat0 + north / 111320.0
    lon = lon0 + east / (111320.0 * math.cos(math.radians(lat0)))
    return lat, lon


# ----------------------------------------------------------- file helpers
def pdf_to_image(raw, dpi=200):
    """Render the first page of a PDF as an RGB image."""
    if fitz is None:
        raise HTTPException(500, "PDF support needs PyMuPDF. Run: pip install pymupdf")
    try:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            if doc.page_count == 0:
                raise HTTPException(400, "The PDF has no pages")
            pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Could not read the PDF file")


def r3(v):
    return round(float(v), 3)


# --------------------------------------------------------------- endpoint
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    row_spacing_ft: float = Form(...),
    plant_spacing_ft: float = Form(...),
    confidence: float = Form(0.25),
):
    if row_spacing_ft <= 0 or plant_spacing_ft <= 0:
        raise HTTPException(400, "Row spacing and plant spacing must be positive")

    raw = await file.read()

    # ---- load the upload (JPG / PNG / PDF)
    if raw[:4] == b"%PDF":
        image = pdf_to_image(raw)
        lat0 = lon0 = yaw = None          # PDFs carry no drone GPS tags
    else:
        try:
            src = Image.open(io.BytesIO(raw))
            src.load()
        except Exception:
            raise HTTPException(400, "Could not read the image file")
        lat0, lon0, yaw = read_gps(raw, src)
        image = ImageOps.exif_transpose(src).convert("RGB")

    W, H = image.size
    rgb = np.array(image)

    # ---- 1. scale from row spacing (feet per pixel)
    est = estimate_row_spacing(rgb)
    if est is None:
        raise HTTPException(
            422, "Could not detect regular crop rows. Use a clearer top-down photo of the rows.")
    row_px = est["spacing_px"]
    ft_per_px = row_spacing_ft / row_px
    m_per_px = ft_per_px * FT_TO_M
    orientation = est["orientation"]

    # ---- 2. detect gaps
    res = model.predict(image, conf=confidence, imgsz=1280, verbose=False)[0]
    boxes = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        for b, c in zip(xyxy, conf):
            boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)))

    # ---- 3. apply spacing rules, number the gaps
    kept, _dropped, _n_merged = finalize_boxes(
        boxes, orientation, ft_per_px, row_spacing_ft, plant_spacing_ft)
    ordered = order_gaps(kept, orientation, row_px)

    # ---- 4. annotated image: red numbered boxes only
    annotated = draw_gaps(image, ordered)
    if annotated.width > 1600:
        r = 1600 / annotated.width
        annotated = annotated.resize((1600, int(annotated.height * r)))
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=85)
    annotated_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    # ---- 5. per-gap results (metres / square metres)
    gaps = []
    lengths_m = []
    used_csv = False
    for i, g in enumerate(ordered, 1):
        x1, y1, x2, y2 = g["box"]
        lat = lon = None
        if lat0 is not None and lon0 is not None:
            lat, lon = pixel_to_latlon((x1 + x2) / 2, (y1 + y2) / 2, W, H,
                                       m_per_px, lat0, lon0, yaw)
        elif i <= len(CSV_GPS):
            lat, lon = CSV_GPS[i - 1]
            used_csv = True

        length_m = g["length_ft"] * FT_TO_M
        lengths_m.append(length_m)
        gaps.append({
            "id": i,
            "lengthM": r3(length_m),
            "widthM": r3(g["width_ft"] * FT_TO_M),
            "areaM2": r3(g["area_ft2"] * FT_TO_M ** 2),
            "plantsNeeded": plants_needed(g["length_ft"], plant_spacing_ft),
            "box": {"x1": x1 / W, "y1": y1 / H, "x2": x2 / W, "y2": y2 / H},
            "lat": lat,
            "lon": lon,
        })

    # ---- 6. field-level numbers
    total_gap_len_m = sum(lengths_m)
    rows = count_rows(W, H, orientation, row_px)
    along_px = H if orientation == "vertical" else W
    row_len_m = along_px * m_per_px
    total_row_len_m = rows * row_len_m
    gap_percent = min(100.0, total_gap_len_m / total_row_len_m * 100) if total_row_len_m else 0.0

    gps_source = "photo" if lat0 is not None else ("csv" if used_csv else None)

    return {
        "gapCount": len(gaps),
        "totalGapLengthM": r3(total_gap_len_m),
        "avgGapLengthM": r3(total_gap_len_m / len(gaps)) if gaps else 0,
        "largestGapM": r3(max(lengths_m)) if lengths_m else 0,
        "totalPlantsNeeded": sum(g["plantsNeeded"] for g in gaps),
        "imageAreaM2": r3(W * H * m_per_px ** 2),
        "rowCount": rows,
        "rowLengthM": r3(row_len_m),
        "totalRowLengthM": r3(total_row_len_m),
        "gapPercent": r3(gap_percent),
        "ftPerPixel": round(ft_per_px, 5),
        "rowSpacingFt": row_spacing_ft,
        "plantSpacingFt": plant_spacing_ft,
        "rowOrientation": orientation,
        "rowReliability": round(est["reliability"], 2),
        "gaps": gaps,
        "annotatedImage": annotated_url,
        "gpsAvailable": any(g["lat"] is not None for g in gaps),
        "gpsSource": gps_source,
        "headingKnown": yaw is not None,
    }