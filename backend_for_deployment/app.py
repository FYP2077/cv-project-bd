from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from models import predict_emotion
from ultralytics import YOLO
import numpy as np
from pydantic  import BaseModel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import torchaudio
import cv2
import os

class AudioData(BaseModel):
    file : str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = '/mnt/data/audiomodel'

model = AutoModelForAudioClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

def predict_emotion(path):
    speech_array, sampling_rate = torchaudio.load(path)

    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    speech = resampler(speech_array).squeeze().numpy()

    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = torch.argmax(logits).item()
    return model.config.id2label[predicted_class_id]

@app.post("/upload")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to the 'audio_data' folder
        file_path = os.path.join('audio_data', file.filename)
        # print(file.filename)
        try:
            uploads = os.listdir('audio_data/')
            for upload in uploads:
            #     if file.filename != upload:
                os.remove(f'audio_data/{upload}')
        except:
            pass        

        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        return { "Erorr" : str(e) }


@app.post("/img_emotion")
async def upload_file(file: UploadFile = File(...)):
    file_bytes = await file.read()

    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Uploaded file is not a valid image"}
    model = 'weights/best.pt'
    model = YOLO(model)
    results = model.predict(img)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = model.names[class_id]
    return {"filename": file.filename, "shape": img.shape, "detection" : label}
    
@app.get('/audio_prediction')
async def audio_prediction():
    audio_path = 'audio_data/'
    label = predict_emotion(f"{audio_path}/{os.listdir(audio_path)[0]}")
    return {'prediction' : label}