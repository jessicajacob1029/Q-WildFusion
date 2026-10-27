from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import uuid

app = FastAPI()

model = YOLO("weights/best.pt")

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    filename = f"tmp_{uuid.uuid4()}.jpg"
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    results = model(filename)
    results[0].save(filename)

    return {"result_image": filename}

