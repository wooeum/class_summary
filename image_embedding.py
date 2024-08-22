import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn


class ImageEmbeddingModel:
    def __init__(self, weight_path, device):
        self.img_width, self.img_height, self.img_channel = 224, 224, 3
        self.device = device
        self.model = self.embedding_model(weight_path)

    def embedding_model(self, weight_file):
        # 모델 초기화  및 가중치 불러오기
        base_model = models.resnet50(weights=None)
        base_model.fc = nn.Linear(2048, 4)
        
        base_model.load_state_dict(torch.load(weight_file, map_location=self.device))
        base_model.eval()

        # 이미지 분류 레이어 제거
        base_model = torch.nn.Sequential(*(list(base_model.children())[:-1]))
        return base_model

    def get_embedding(self, img_path):
        # 이미지 로드
        img = Image.open(img_path).convert('RGB')

        # 이미지 변환 정의
        preprocess = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 이미지 전처리
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # 배치 차원 추가 (1, 3, 224, 224)
        # img_tensor = img_tensor.to(device)
        # 모델을 평가 모드로 설정
        self.model.eval()

        with torch.no_grad():
            # 예측 및 1차원 배열로 변환
            embedding = self.model(img_tensor)

        return embedding.view(-1)

def main(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dict = {"onepiece":r"C:\Users\USER\Desktop\onepiece_epoch50\049.pth"}
    
    emb_model = ImageEmbeddingModel(weight_dict['onepiece'], "cpu")
    emb_vec = emb_model.get_embedding(img_path)

    return emb_vec

if __name__ == "__main__":
    test_img = r"C:\Users\USER\Pictures\Screenshots\스크린샷 2024-07-17 160254.png"
    embeddings = main(test_img)
    print(embeddings.shape)