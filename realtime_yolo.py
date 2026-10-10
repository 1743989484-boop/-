from ultralytics import YOLO

# 可以切换为自己训练出来的best.pt
# model = YOLO("runs/train/coco128_exp/weights/best.pt")
model = YOLO("yolov8n.pt")

if __name__ == "__main__":
    # source=0 本地摄像头；写视频路径就是读视频文件
    result_stream = model(source=0, stream=True, show=True)

    for frame_result in result_stream:
        box_list = frame_result.boxes
        obj_names = [model.names[int(b.cls)] for b in box_list]
        print(f"本帧检测到物体：{obj_names}")