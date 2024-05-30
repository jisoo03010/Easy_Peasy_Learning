from flask import Flask, render_template, request, jsonify
from mobilenet_img_model import  train_model, load_data , predict
import time

app = Flask(__name__)

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



@app.route('/train/parameters', methods=['POST'])
def save_training_parameters(): 
    args_dict =request.args.to_dict()
    batch_size = args_dict['batch_size']
    num_workers = args_dict['num_workers']
    datasets_path = args_dict['datasets_path']
    device = args_dict['device']
    epochs = args_dict['epochs']
    early_stop_counter = args_dict['early_stop_counter']
    lr = args_dict['lr']
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_save_path = f"./mobilenet_v{timestamp}.pt"

    
    train_dataloaders, val_dataloaders = load_data(batch_size, num_workers, datasets_path)
    model, train_loss, test_accu, model_save_path = train_model(train_dataloaders,val_dataloaders, device, epochs, early_stop_counter, lr, model_save_path)
    
    data = {"model_save_path":model_save_path, "model":model, "train_loss" : train_loss, "test_accu": test_accu}
    return jsonify(data)

    

@app.route("/upload", methods=['POST'])
def upload():
    
    file_upload = request.files['file_lo'] # 받아온 파일 
    pp = predict(file_upload)
    for p in pp:
        predd = {"eum_data" : "{:.4f}%".format(p[0]*100), "ma_data" : "{:.4f}%".format(p[1]*100), "son_data" : "{:.4f}%".format(p[2]*100), "max_data" : "{}".format(pp.argmax(1).item() )}
    return jsonify(predd)

    
    


#애플리케이션 실행
if __name__ == '__main__':
    app.run(debug=True)