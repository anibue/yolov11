# Auto-annotate images using YOLO and SAM models
from ultralytics.data.annotator import auto_annotate

auto_annotate(data="D:\\anibue\\yolov11\\env-test\\bus.jpg", det_model="D:\\anibue\\yolov11\\yolo11n-pose.pt", sam_model="D:\\anibue\\yolov11\\sam2_b.pt")
