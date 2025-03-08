from ultralytics import YOLO
def main():
    model = YOLO('yolo11s-cls.pt')
    # Start training
    model.train(data="augmented_waste_classification", epochs=50, imgsz=224)
if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()