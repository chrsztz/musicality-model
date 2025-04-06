from ultralytics import YOLO
import multiprocessing


def main():
    # Load your trained model
    model = YOLO('yolo11s-cls.pt')  # or path to your trained model

    # Validate the model
    metrics = model.val(data="dataset-original")

    # For classification models, use the appropriate metrics
    print(f"Top-1 Accuracy: {metrics.top1}")
    print(f"Top-5 Accuracy: {metrics.top5}")
    print(f"Model Fitness: {metrics.fitness}")

    # If you want to see all available metrics
    print("All metrics:")
    print(metrics.results_dict)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()