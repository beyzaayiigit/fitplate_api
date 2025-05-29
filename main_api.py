from fastapi import FastAPI, UploadFile, File
from worker_api import detect_food
import uuid
import shutil
import os

app = FastAPI()

# Geçici klasör oluştur (ilk başlatmada)
os.makedirs("temp_images", exist_ok=True)

@app.post("/predict/")
def predict(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = f"temp_images/{file_id}.jpg"

    # Dosyayı sunucuya kaydet
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Celery task'ını çağır
    task = detect_food.delay(file_path)
    return {"task_id": task.id}

@app.get("/result/{task_id}")
def get_result(task_id: str):
    result = detect_food.AsyncResult(task_id)
    if result.state == "PENDING":
        return {"status": "PENDING"}
    elif result.state == "FAILURE":
        return {"status": "FAILURE"}
    elif result.state == "SUCCESS":
        return {"status": "SUCCESS", "result": result.result}
    else:
        return {"status": result.state}
