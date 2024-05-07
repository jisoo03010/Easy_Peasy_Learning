
# -*- coding: utf-8 -*-
import argparse
import base64
import io
import logging
import torch
import torchvision.transforms as transforms
from flask import Flask, request , render_template
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from torchvision.models import mobilenet_v2


app = Flask(__name__)
cors = CORS (app, resources={r"/predict/*" : {"origins" : "*"}})

# 이미지 처리 메서드 
def img_transform(image_bytes):
    data_transforms = transforms.Compose([
        transforms.Resize((230, 230)),
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(p=1.0), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = data_transforms(image)
    image = image.unsqueeze(0)
    image = image.reshape(1, 3, 230, 230) # 배치, 채널, 높이, 너비
    
    return image


# boxes > json 배열 생성 메서드  
def boxes_to_json(tensor_img_size):
    try:
        result = [] 
        if len(tensor_img_size) != 0:
            result.append({"x": 0, "y":0, "w" : tensor_img_size[2], "h" :tensor_img_size[3], "gender" : "", "age" : ""})
    except IndexError as e:
        logging.error(e)
    return result


# html view에 표시될 이미지 형변환 메서드
def serve_pil_image(pil_img):
    try:
        img_io = io.BytesIO()
        pil_img.save(img_io, 'JPEG', quality=70)
        img_io.seek(0)
        image = base64.b64encode(img_io.read()).decode("utf-8")
        return image
    except Exception as e:
        logging.error(e)
        return None

def model_load(model, model_path):
    if DEVICE == "cpu":
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        model.eval()
        model = model.to(DEVICE)
    return model

#  parameter : image file data && request_image를 받는다. 
@app.route('/inference/binary_classification', methods=['POST'])
def binary_classification():
    result_code = "200"
    result_message = "OK"
    
    is_request_image = request.form.get("request_image")
    try:
        image_data = request.files['image'].read()  
        
        image_tensor = img_transform(image_data)  
        frame = Image.open(io.BytesIO(image_data))

    except UnidentifiedImageError:
        result_code = "400" 
        result_message = "Cannot identify image file. That is bad image."
        return { "result_code": result_code, "result_message": result_message }

    
    try:
        
        pred  = MODEL(image_tensor.to(DEVICE))
        data = torch.sigmoid(pred).item() 
        
        human_data = (1 -  data ) * 100
        no_human_data = data * 100
        print(f"human_data : {human_data:.4f}")
        print(f"no_human_data : {no_human_data:.4f}")
        if human_data > no_human_data:
            json_boxes = []
            json_boxes = boxes_to_json(image_tensor.size())
        else:
            json_boxes = []

        result = { "result_code": result_code, "result_message": result_message, "data": { 'boxes':  json_boxes} }

        if is_request_image :
            result['data']['image'] = "data:image/jpeg;charset=utf-8;base64," + serve_pil_image(frame)
    
    except FileNotFoundError as e:
        logging.error(e)
        result = {}

    return result


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect faces in image.")
    parser.add_argument('--port', default=8080, help = "Server port : default 8080")
    parser.add_argument('--enable-ssl', action='store_true', help = "Enable ssl : default false")
    parser.add_argument('--model-path', default="./mobilenet_v6.pt", help="Path to the model file")
    params = parser.parse_args()
    server_port = params.port
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    MODEL = mobilenet_v2(pretrained=False, num_classes=1)
    MODEL = model_load(MODEL , params.model_path)
    enable_ssl =  'adhoc' if params.enable_ssl else None
    app.run(port=server_port, host='0.0.0.0', ssl_context=enable_ssl)