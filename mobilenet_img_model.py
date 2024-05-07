import argparse
import os 
import time
import torch
import torch.nn as nn
import logging
import torchvision.datasets as datasets_module
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from torchvision.models import mobilenet_v2 
from tqdm import tqdm
import shutil
import splitfolders 


best_loss = float('inf')
num_workers = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

# model 초기화 함수 
def initialize_model():
    model = mobilenet_v2(pretrained=False, num_classes=1)
    return model



def test_load_data(batch_size, num_workers, datasets_path):
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((230, 230)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    
    datasets = {x: datasets_module.ImageFolder(os.path.join(datasets_path, x), data_transforms[x])
            for x in ['test']}

    test_dataloaders = torch.utils.data.DataLoader(
            datasets['test'],
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
    )
    
    return test_dataloaders
    
    
def load_data(batch_size, num_workers, datasets_path):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((230, 230)),
            transforms.ColorJitter(brightness=(0.8, 0.9)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((230, 230)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    datasets = {x: datasets_module.ImageFolder(os.path.join(datasets_path, x), data_transforms[x])
            for x in ['train', 'val']}

    train_dataloaders = torch.utils.data.DataLoader(
        datasets['train'],
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_dataloaders = torch.utils.data.DataLoader(
        datasets['val'],
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloaders, val_dataloaders


def train_model(train_dataloaders,val_dataloaders, device, epochs, early_stop_counter, lr, model_save_path):
    
   
    model = initialize_model() 
    
    loss_func = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
 
    model.to(device)
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        loss_eq = 0
        train_loss = []
        test_loss = [] 
        train_accu = []
        test_accu = []
        
        y_arr = [] 
        pred_arr = []
        print(f"Epoch {epoch+1}\n-------------------------------") 
        for x, y in tqdm(train_dataloaders):
            x =  x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_func(pred, y.float().unsqueeze(1))

            y_arr += y.tolist()
            pred_arr += (pred > 0.5).tolist()

            
            
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            torch.save( model.state_dict(), model_save_path)
            loss_eq += loss.item() 

        f1 = f1_score(y_arr, pred_arr, average='macro')
        s= (len(train_dataloaders) * epochs) / args.batch_size
        print(f"Step : {s}")
        print(f"train loss : {(loss_eq / len(train_dataloaders)):.4f}, train accuracy : {(f1 * 100):.4f}% ")
        train_loss.append( loss_eq / len(train_dataloaders) )
        train_accu.append(f1)

        model.eval()
        loss_eq = 0
        epoch_time = time.time() - start_time
        y_arr = [] 
        pred_arr = []
        with torch.no_grad():
            for x, y in val_dataloaders:
                x = x.to(device)
                y = y.to(device)
                pred = model(x) 

                y = y.float().unsqueeze(1)
                    
                y_arr += y.tolist()
                pred_arr += (pred > 0.5).tolist()
                
                loss = loss_func(pred, y)
                loss_eq += loss.item()

            f1 = f1_score(y_arr, pred_arr, average='macro')
            print(f"val loss : {(loss_eq / len(val_dataloaders)):.4f}, val accuracy : {(f1 * 100):.4f}% ")

            test_loss.append(loss_eq / len(val_dataloaders))
            test_accu.append(f1)

            if loss_eq > best_loss:
                early_stop_counter += 1
            else:
                best_loss = loss_eq
                early_stop_counter = 0

            print(f"early_stop_counter : {early_stop_counter}")
            print(f"Epoch Time: {epoch_time:.2f} seconds \n\n")

            # early stop 
            if early_stop_counter >= args.early_stopping_epochs:
                logging.info("Early Stopping!")
                break 
    return model, train_loss, test_accu


def test_model(test_dataloaders, device, model_file_path):
        
    model = initialize_model() 
    if device == "cpu":
        model.load_state_dict(torch.load(model_file_path))
    else:    
        model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))

    model.to(device)
    model.eval()

    total_cnt = 0
    y_arr_test = []
    pred_arr_test = []
    with torch.no_grad(): 
        for x, y in tqdm(test_dataloaders):
            
            y = y.to(device)
            x = x.to(device) 
            pred = model(x) # 모델에 x 넣어서 예측한 출력값 
            y = y.float().unsqueeze(1)
            total_cnt += y.size(0) # 정수로 변환하여 누적

            y_arr_test += y.tolist()
            pred_arr_test += (pred > 0.5).tolist()
            
        f1 = f1_score(y_arr_test, pred_arr_test, average='macro')
        f1_last_data = f1 * 100
        print(f"Test Accuracy f1 : {(f1_last_data):.4F}% \nTotal_cnt : {int(total_cnt)}")
    return f1_last_data



if __name__ == "__main__":
    
    # 인수 설정
    parser = argparse.ArgumentParser(description='How to use Argument')
    parser.add_argument('--epoch', type=int, default=60, help='epoch setting')
    parser.add_argument('--early-stopping-epochs', type=int, default=5, help='early stopping epochs number setting')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size setting')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    
    parser.add_argument('--datasets-path', type=str, default= "./sample_data") # 데이터셋의 위치 경로 
    
    parser.add_argument('--zip-folder-path', type=str, default= './dataset.zip')
    
    # 저장할 모델의 파일이름 지정
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    default_model_save_path = f"./mobilenet_v{timestamp}.pt"
    parser.add_argument('--model-save-path', type=str, default=default_model_save_path, help='Path to save the model') # 모델 저장시 사용하는 경로
    parser.add_argument('--model-file-path', type=str, default="./mobilenet_v6.pt", help='The path of the .pt file used to load the model') # 모델 로드시 사용하는 .pt 파일 경로

    parser.add_argument('--test', action='store_true') # 테스트 데이터 로드여부 지정 
    args = parser.parse_args()

    # 'sample_data' 폴더의 존재를 확인
    if os.path.isdir(args.datasets_path) == False:
        print("args.datasets_path : ", args.datasets_path )
        
        shutil.unpack_archive(args.zip_folder_path,'./sample_data' ,'zip')
        splitfolders.ratio("./sample_data/dataset/", output='./sample_data/', seed=1337, ratio=(0.6, 0.2, 0.2))

    # '--test' 옵션을 사용하면 모델이 테스트 데이터로 실행
    # Or args.test = false -> 학습데이터로 실행된다.
    if args.test == True:
        test_dataloaders = test_load_data(args.batch_size, num_workers , args.datasets_path) # test할때 사용되는 인자 값 : 배치사이즈, 넘버웍스, 데이터셋 경로지정, 생성한 모델 위치 경로 지정,
        f1_last_data = test_model(test_dataloaders, device, args.model_file_path)
    else:
        train_dataloaders, val_dataloaders = load_data(args.batch_size, num_workers, args.datasets_path)
        model, train_loss, test_accu = train_model(train_dataloaders, val_dataloaders, device, args.epoch, args.early_stopping_epochs , args.lr, args.model_save_path)

