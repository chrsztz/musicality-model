from ultralytics import YOLO
model = YOLO("yolo11s-cls.pt")
model.train(data="augmented_waste_classification", epochs=50, imgsz=224, device="mps")
model.export(format="engine", device="mps", imgsz=224, half=True)  # FP16