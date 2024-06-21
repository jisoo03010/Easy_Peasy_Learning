from flask import Flask, render_template, request, jsonify
from questionary import form
from mobilenet_img_model import  train_model, load_data , predict, test_load_data, test_model
import time
import os
import json    
import torch
import matplotlib.pyplot as plt
app = Flask(__name__)


# 프로그램의 디버깅이 모두 끝나면 웹 애플리케이션에서 디버깅 모드를 끄는게 좋다.
# eu : 디버깅 콘솔에 출력되는 정보를 이용할 경우 외부 공격자에게 웹 애플리케이션의 취약점을 노출할 수 있기 때문이다. 
# app.debug = True

# app.config.update(
#     DEBUG=True


@app.route('/info')
def info_template():
    return render_template("info.html")

@app.route('/project')
def project_template():
    return render_template("project.html")

@app.route('/learning')
def learning_template():
    return render_template("learning.html")

@app.route('/setting')
def setting_template():
    return render_template("setting.html")


@app.route('/')
def index_template():
    return render_template("index.html")



# - 모델 만들기
# 1. 모델 파라미터 입력
# 2. 모델 파이썬 파일.함수에 파라미터 값 전달하기 

# 4. 파일 저장 (모델 pt 파일)

# 3. 모델 파일명 , 결과 정확도, model 값 response 전달하기 
# 4. 화면에서 보여주기 ( 
#   1. 파일명 받아서 프로젝트에 리스트 추가 
#   2. 파일명 받아서 프로젝트에 리스트 추가 

# @app.route('/testAPI', methods=['POST'])
# def test_api_method():
#     args_dict = request.json
#     print(f"args_dict : {args_dict}")
#     return "data"


@app.route('/train/parameters', methods=['POST'])
def save_training_parameters(): 
    data_json = request.get_json()
    print(type(data_json))
    print(data_json)

    model_name = data_json['model_name'] # type: ignore
    batch_size = int(data_json['batch_size']) # type: ignore
    learning_rate = float(data_json['lr'])# type: ignore
    early_stopping_epochs = int(data_json['stop_number'])# type: ignore
    file_name = data_json['file_name']# type: ignore
    epochs = int(data_json['epochs'])# type: ignore
    num_workers = int(data_json['num_workers'])# type: ignore

    print(f"Model Name: {model_name}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Stop Number: {early_stopping_epochs}")
    print(f"File Name: {file_name}")
    print(f"epochs: {epochs}")
    print(f"num_workers: {num_workers}")
    # print(data.get('model_name'))
    # batch_size = data['batch_size']
    # num_workers = data['num_workers']
    datasets_path = "./sample_data"
    zip_folder_path = "./datasets.zip"
    # num_workers = 0
    # epochs = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    # device = data['device']
    # epochs = data['epochs']
    # early_stop_counter = data['early_stop_counter']
    # lr = data['lr']
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_save_path = f"./mobilenet_v{timestamp}.pt"
    print("zip_folder_path ; ;; ; ; :", zip_folder_path)
    print("datasets_path ; ;; ; ; :", datasets_path)
    train_dataloaders, val_dataloaders = load_data(batch_size, num_workers, datasets_path, zip_folder_path)
    model, train_loss, test_accu, model_save_path = train_model( early_stopping_epochs, batch_size, train_dataloaders,val_dataloaders, device, epochs, learning_rate, model_save_path)
    return_datas = {"model_save_path":model_save_path, "train_loss" : train_loss, "test_accu": test_accu}
    return jsonify(return_datas)


@app.route('/test/parameters', methods=['POST'])
def test_parameters():  
    data_json = request.get_json()
    print(type(data_json))
    print(data_json)

    model_pt_file_name = data_json['model_name']# type: ignore # 받아온 파일 
    batch_size = int(data_json['batch_size']) # type: ignore
    num_workers = int(data_json['num_workers'])# type: ignore
    datasets_path = "./sample_data"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    test_dataloaders =  test_load_data(batch_size, num_workers, datasets_path)
    model, f1_last_data,pred_arr_test =  test_model(test_dataloaders, device, model_pt_file_name) # type: ignore
    # grap(pred_arr_test)
    return_datas = {"f1_last_data":f1_last_data}
    return jsonify(return_datas)
    



# @app.route("/viewGraf", methods=['POST']) # type: ignore
# def grap(pred_arr_test):
#     # 그래프 그리기
#     # plt.plot([t for t in torch.as_tensor(train_accu)], label='train accu')
#     plt.plot([t for t in torch.as_tensor(pred_arr_test)], label='test accu')
#     plt.legend()
#     # 이미지 저장
#     plt.savefig('accuracy_plot.png')


@app.route("/upload", methods=['POST'])
def upload():

    file_upload = request.files['file_lo'] # 받아온 파일 
    model_pt_file_name_bolb = request.files['model_pt_file_name']# 받아온 파일 
    model_pt_file_name = json.load(model_pt_file_name_bolb) # type: ignore
    print('file_upload :', model_pt_file_name)
    # model_pt_file_name = jsonify(model_pt_file_name_blob) # type: ignore
    # model_pt_file_name =  upload_model_name() 
    # model_pt_file_name = upload_model_name().json['model_name'] # type: ignore
    print('model_pt_file_name :', model_pt_file_name)
    pp = predict(file_upload ,model_pt_file_name)
    print("pp =======> " , pp)
    # for p in pp:
    #     predd = {"eum_data" : "{:.4f}%".format(p[0]*100), "ma_data" : "{:.4f}%".format(p[1]*100), "son_data" : "{:.4f}%".format(p[2]*100), "max_data" : "{}".format(pp.argmax(1).item() )}
    return pp


#  파일 목록 가져오기 
@app.route("/fileList", methods=['GET']) # type: ignore
def file_list():
    pt_files = [f for f in os.listdir() if os.path.isfile(f) and f.endswith('.pt')]
    # 파일 이름 출력
    for file in pt_files:
        print(file)
    return jsonify(pt_files)


#애플리케이션 실행
if __name__ == '__main__':
    app.run(debug=True)