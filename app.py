from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import base64

app = FastAPI()

# allow Express backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model once when server starts
model = YOLO("best.pt")
# this model's classes are {0: "GAP", 1: "PLANT"}

GAP_COLOR = (139, 0, 0)     # dark red — used for gap boxes
PLANT_COLOR = (0, 70, 0)    # dark green — only used if DRAW_PLANT_BOXES is True
BOX_WIDTH = 4

# set to True if you also want plant detections boxed on the image
DRAW_PLANT_BOXES = False


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image)
    boxes = results[0].boxes
    class_names = results[0].names  # {0: "GAP", 1: "PLANT"}

    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    gap_count = 0
    plant_count = 0

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id].upper()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        if cls_name == "GAP":
            gap_count += 1
            color = GAP_COLOR
            label = f"Gap {conf:.2f}"
        elif cls_name == "PLANT":
            plant_count += 1
            if not DRAW_PLANT_BOXES:
                continue
            color = PLANT_COLOR
            label = f"Plant {conf:.2f}"
        else:
            continue

        draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_WIDTH)

        text_bbox = draw.textbbox((x1, y1), label)
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - text_h - 4)

        draw.rectangle(
            [x1, label_y, x1 + (text_bbox[2] - text_bbox[0]) + 6, label_y + text_h + 4],
            fill=color,
        )
        draw.text((x1 + 3, label_y + 2), label, fill=(255, 255, 255))

    total_plants = plant_count
    total_positions = gap_count + total_plants
    gap_percentage = (
        round((gap_count / total_positions) * 100, 1) if total_positions > 0 else None
    )

    buffer = io.BytesIO()
    annotated_image.save(buffer, format="JPEG", quality=90)
    annotated_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "gapCount": gap_count,
        "totalPlants": total_plants,
        "gapPercentage": gap_percentage,
        "annotatedImage": f"data:image/jpeg;base64,{annotated_base64}",
    }