from ultralytics import YOLO

# 加载预训练权重作为迁移学习
model = YOLO("yolov8n.pt")

if __name__ == "__main__":
    # coco128是128张图片小数据集，适合练手
    results = model.train(
        data="coco128.yaml",
        epochs=10,          # 训练轮次，10轮很快
        batch=8,
        imgsz=640,
        device="cpu",       # 有N卡写0，cpu就写cpu
        project="runs/train",
        name="coco128_exp"
    )
    print("训练完成！权重保存在 runs/train/coco128_exp/weights/best.pt")