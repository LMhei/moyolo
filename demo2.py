from ultralytics import YOLO

#model = YOLO('ultralytics/cfg/models/mo_yolov8.yaml')
model = YOLO('ultralytics/cfg/models/v8/yolov8s.yaml')
model.info()  # 打印模型结构信息
print(model.model)

