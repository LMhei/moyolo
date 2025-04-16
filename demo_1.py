from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train():
    # model = YOLO("./ultralytics/cfg/models/v8/mtyolov8.yaml")
    model = YOLO("./yolov8n.pt")
    model.train(data="./ultralytics/cfg/datasets/mt.yaml", epochs=20)
    result = model.val()

def my_pre():
    model = YOLO("./best.pt")
    # 加载大图
    large_image = cv2.imread('./mt.jpg')
    # 定义分割参数
    segment_width = 640
    segment_height = 640
    stride = 640  # 或其他你希望的步长
    color_map = {
        'hole': (255, 0, 0),
        'crack': (0, 255, 0)
        # 其他类别的颜色可以根据需要添加
    }

    for y in range(0, large_image.shape[0], stride):
        for x in range(0, large_image.shape[1], stride):
            # 提取分割区域
            segment = large_image[y:y + segment_height, x:x + segment_width]
            # 对分割区域进行目标检测
            results = model.predict(segment,conf=0.5)
            for result in results:
                boxes = result.boxes
                names = result.names
                num = len(boxes.cls.cpu().numpy().astype(int))
                if num >= 1:
                    for i in range(num):
                        xyxy = boxes.xyxy.cpu().numpy().astype(int)[i]
                        cls = boxes.cls.cpu().numpy().astype(int)[i]
                        conf = boxes.conf.cpu().numpy()[i]
                        color = color_map.get(names.get(cls), (0, 255, 0))  # 默认绿色
                        # 将RGB格式的颜色转换为BGR格式
                        color = (color[2], color[1], color[0])
                        x1 = xyxy[0] + x
                        y1 = xyxy[1] + y
                        x2 = xyxy[2] + x
                        y2 = xyxy[3] + y
                        cv2.rectangle(large_image, (x1, y1), (x2, y2), color, 4)
                        cv2.putText(large_image, f"{names.get(cls)} {conf:.2f}", (x1, y1 - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    cv2.imwrite("./result1.jpg",large_image)
    plt.title('Pre')
    plt.imshow(cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':
     model = YOLO("mtyolov8.yaml")
     model.train(data="./ultralytics/cfg/datasets/mt.yaml", epochs=50)
    #my_pre()
    # model = YOLO("./best.pt")
    # results = model("./mt.jpg")
    # for result in results:
    #     result.show()
    #     result.save("./result.jpg")
