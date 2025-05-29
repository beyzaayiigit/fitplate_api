from celery import Celery
from ultralytics import YOLO
import cv2
import os

app = Celery(
    "worker_api",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

model = YOLO("food_best.pt")

# Besin değerleri sözlüğü (100 gram)
nutrition_info = {
    "bakla": {"kalori": 47, "yağ": 1.0, "protein": 3.6, "karbonhidrat": 4.4},
    "baklava": {"kalori": 428, "yağ": 29.03, "protein": 6.7, "karbonhidrat": 37.62},
    "cikolatali pasta": {"kalori": 371, "yağ": 16.0, "protein": 4.0, "karbonhidrat": 53.0},
    "donut": {"kalori": 452, "yağ": 25.0, "protein": 4.9, "karbonhidrat": 51.0},
    "et": {"kalori": 250, "yağ": 15.0, "protein": 26.0, "karbonhidrat": 0.0},
    "hamburger": {"kalori": 165, "yağ": 6.12, "protein": 7.02, "karbonhidrat": 19.88},
    "haslanmis yumurta": {"kalori": 155, "yağ": 10.61, "protein": 12.58, "karbonhidrat": 1.12},
    "havuclu kek": {"kalori": 389, "yağ": 17.0, "protein": 4.0, "karbonhidrat": 56.0},
    "kapkek": {"kalori": 389, "yağ": 17.0, "protein": 4.0, "karbonhidrat": 56.0},
    "kore mantisi": {"kalori": 170, "yağ": 3.5, "protein": 4.1, "karbonhidrat": 29.7},
    "midye": {"kalori": 86, "yağ": 0.96, "protein": 14.67, "karbonhidrat": 3.57},
    "omlet": {"kalori": 154, "yağ": 11.0, "protein": 11.0, "karbonhidrat": 1.0},
    "pankek": {"kalori": 227, "yağ": 7.0, "protein": 6.0, "karbonhidrat": 35.0},
    "patates kizartmasi": {"kalori": 312, "yağ": 14.73, "protein": 3.43, "karbonhidrat": 41.44},
    "peynir tabagi": {"kalori": 402, "yağ": 33.0, "protein": 25.0, "karbonhidrat": 1.3},
    "pizza": {"kalori": 265, "yağ": 5.0, "protein": 7.0, "karbonhidrat": 48.0},
    "red velvet": {"kalori": 423, "yağ": 21.0, "protein": 4.0, "karbonhidrat": 55.0},
    "salata": {"kalori": 30, "yağ": 1.0, "protein": 0.9, "karbonhidrat": 4.7},
    "sandvic": {"kalori": 160, "yağ": 6.54, "protein": 7.39, "karbonhidrat": 17.56},
    "sarimsakli ekmek": {"kalori": 350, "yağ": 15.0, "protein": 7.0, "karbonhidrat": 45.0},
    "sogan halkasi": {"kalori": 411, "yağ": 22.0, "protein": 4.0, "karbonhidrat": 49.0},
    "soslu makarna": {"kalori": 89, "yağ": 2.2, "protein": 3.3, "karbonhidrat": 13.71},
    "spagetti": {"kalori": 158, "yağ": 1.0, "protein": 6.0, "karbonhidrat": 31.0},
    "tavuk kanat": {"kalori": 203, "yağ": 13.0, "protein": 19.0, "karbonhidrat": 0.0},
    "waffle": {"kalori": 291, "yağ": 14.1, "protein": 7.9, "karbonhidrat": 32.9}
}

@app.task
def detect_food(image_path):
    try:
        image = cv2.imread(image_path)

        if image is None:
            return {"error": "Image could not be read"}

        results = model(image)

        if not results or not hasattr(results[0], "boxes"):
            return {"warning": "No detection results"}

        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                nutrition = nutrition_info.get(label, "bilgi bulunamadı")

                detections.append({
                "yemek": label,
                "doğruluk": round(conf, 2),
                "besin değeri": nutrition
            })
                
        # Görseli işledikten sonra sil
        try:
            os.remove(image_path)
        except:
            pass

        return detections if detections else {"warning": "No objects detected"}

    except Exception as e:
        return {"error": str(e)}
