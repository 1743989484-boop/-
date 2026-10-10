from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO
import io
import requests

app = FastAPI(title="Visual Harness Service")
app.mount("/static", StaticFiles(directory="static"), name="static")

model = YOLO("yolov8n.pt")

# 原有接口：接收上传文件，供网页前端使用
@app.post("/detect")
async def detect(file: UploadFile):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
        res = model(img)
        obj_names = set()
        for r in res:
            for box in r.boxes:
                cls_id = int(box.cls)
                obj_name = r.names[cls_id]
                obj_names.add(obj_name)
        return {"status":"ok","found_objects": list(obj_names)}
    except Exception as e:
        return {"status":"error","msg":f"图片检测失败：{str(e)}"}

# 新增接口：接收图片url，适配Dify
@app.post("/detect_by_url")
async def detect_by_url(img_url: str):
    try:
        resp = requests.get(img_url,timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        res = model(img)
        obj_names = set()
        for r in res:
            for box in r.boxes:
                cls_id = int(box.cls)
                obj_name = r.names[cls_id]
                obj_names.add(obj_name)
        return {"status":"ok","found_objects": list(obj_names)}
    except Exception as e:
        return {"status":"error","msg":f"url图片检测失败：{str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001)