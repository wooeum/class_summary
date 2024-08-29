import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import chromadb
import os
import sys
import django

# Django 설정 초기화
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'GoodMarket.settings')
django.setup()

from market.models import ProductFile

class ImageEmbeddingModel:
    def __init__(self, weight_path, class_num, img_path):
        # self.img_width, self.img_height, self.img_channel = 224, 224, 3
        self.emb_model = self.create_model(weight_path, class_num, embedding=True)
        self.csf_model = self.create_model(weight_path, class_num, embedding=False)
        self.img_tensor = self.preprocess_image(img_path)

    def create_model(self, weight_file, class_num, embedding):
        base_model = models.resnet50(weights=None)
        
        base_model.fc = nn.Linear(2048, class_num)
        base_model.load_state_dict(torch.load(weight_file, map_location='cpu', weights_only=True))
        base_model.eval()
        if embedding:
            base_model = torch.nn.Sequential(*(list(base_model.children())[:-1]))
        return base_model
    
    def preprocess_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            
        ])
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor
    
    def get_embedding(self):
        with torch.no_grad():
            embedding = self.emb_model(self.img_tensor)
        return embedding.view(-1)

    def get_category(self):
        with torch.no_grad():
            category = self.csf_model(self.img_tensor)
        return category


def get_chromadb_collection(collection_name="product_images", db_path="./chroma_db_data"):
    client = chromadb.PersistentClient(path=db_path)
    try:
        return client.get_collection(name=collection_name)
    except ValueError as e:
        if 'does not exist' in str(e):
            print(f"Collection '{collection_name}' does not exist. Creating new collection.")
            return client.create_collection(name=collection_name)
        raise ValueError(f"ChromaDB collection error: {str(e)}")

def embed_and_store_product_images(cate_big, model_info_dict):
    images_collection = get_chromadb_collection()
    # 모든 ProductFile 객체 가져오기
    product_files = ProductFile.objects.all()
    
    for product_file in product_files:
        img_path = product_file.file.path
        cate_big = cate_big
        # 임베딩 추출 및 ChromaDB에 저장
        try:
            base_model = ImageEmbeddingModel(
                weight_path=model_info_dict[cate_big][0],
                class_num=model_info_dict[cate_big][1],
                img_path=img_path
            )
            emb_vec = base_model.get_embedding()
            category = base_model.get_category()

            image_id = str(product_file.product_file_id)  # 필드명 `product_file_id`로 수정
            embedding_vector = emb_vec.tolist()

            # ChromaDB에 임베딩과 메타데이터 추가
            images_collection.add(
                embeddings=[embedding_vector],
                ids=[image_id], 
                metadatas=[{"product_file_id": image_id}]
            )

            max_value, max_index = torch.max(category, dim=1)
            print(f"Embedding size: {emb_vec.shape}, Category: {max_index.item()}")
            print(f"Image {image_id} embedding has been stored in chromaDB.")

        except Exception as e:
            print(f"Error processing image {product_file.product_file_id}: {e}")

def find_similar_images(img_path, cate_big, model_info_dict, top_k=8):
    base_model = ImageEmbeddingModel(
        weight_path=model_info_dict[cate_big][0],
        class_num=model_info_dict[cate_big][1],
        img_path=img_path
    )
    embed_and_store_product_images(cate_big, model_info_dict)


    new_emb_vec = base_model.get_embedding()
    images_collection = get_chromadb_collection()

    stored_data = images_collection.get(include=['embeddings', 'metadatas'])

    if not stored_data or not stored_data.get('embeddings'):
        print("Error: No embeddings found in ChromaDB.")
        return []

    similarities = []
    new_emb_vec = new_emb_vec.unsqueeze(0)

    for idx, stored_emb in enumerate(stored_data['embeddings']):
        stored_emb_tensor = torch.tensor(stored_emb).unsqueeze(0)
        similarity = torch.nn.functional.cosine_similarity(new_emb_vec, stored_emb_tensor)
        similarities.append((similarity.item(), stored_data['metadatas'][idx]['product_file_id']))

    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:top_k]  # 리스트를 반환