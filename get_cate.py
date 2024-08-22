import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

img_width, img_height, img_channel = 224, 224, 3

def embedding_model(weight_file):
    # 모델 초기화  및 가중치 불러오기
    base_model = models.resnet50(weights=None)
    base_model.fc = nn.Linear(2048, 4)
    
    base_model.load_state_dict(torch.load(weight_file, map_location='cpu'))
    base_model.eval()
    return base_model

def get_category(img_path, base_model):
    img = Image.open(img_path).convert('RGB')

    # 이미지 변환 정의
    preprocess = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 이미지 전처리
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # 배치 차원 추가 (1, 3, 224, 224)
    # img_tensor = img_tensor.to(device)
    # 모델을 평가 모드로 설정
    base_model.eval()
    cate = base_model(img_tensor)

    return cate


test_img = r"C:\DA35_Project\final_project\dataset\yolo_dataset\modify_data\images\506064.jpg"    
weight_dict = {"onepiece":r"C:\Users\USER\Desktop\onepiece_epoch50\049.pth"}
cate = get_category(test_img, embedding_model(weight_dict['onepiece']))

print(cate)

