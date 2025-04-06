from ultralytics import YOLO

model = YOLO('best1.pt')
model.export(format = 'coreml')