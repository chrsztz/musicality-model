from ultralytics import YOLO
def main():
    model = YOLO('yolo11s-cls.pt')
    # Start training
    model.train(data="dataset-original", epochs=20, imgsz=256)
if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()