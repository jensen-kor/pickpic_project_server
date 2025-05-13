import os
import torch
from PIL import Image
import base64
import numpy as np
from typing import Optional, List

def find_similar_images_by_clip(text: str, image_dir: str, features_dir: str, top_n: int = 5) -> Optional[List[dict]]:
    """
    미리 저장된 이미지 임베딩(features_dir/*.npy)과 입력 텍스트 임베딩을 비교하여
    유사도가 높은 이미지 top_n개를 반환 (base64, 파일명)
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        raise ImportError('transformers, torch 패키지가 필요합니다.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    # 텍스트 임베딩 추출
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs).cpu().numpy()[0]

    # 이미지 임베딩(.npy) 파일 목록
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
    if not feature_files:
        return None

    similarities = []
    for feat_file in feature_files:
        img_feature = np.load(os.path.join(features_dir, feat_file))
        # 코사인 유사도 계산
        sim = np.dot(text_features, img_feature) / (np.linalg.norm(text_features) * np.linalg.norm(img_feature))
        similarities.append((feat_file.replace('.npy', ''), sim))

    # 유사도 내림차순 정렬 후 top_n개 선택
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_files = similarities[:top_n]

    results = []
    for fname, score in top_files:
        img_path = os.path.join(image_dir, fname)
        if not os.path.exists(img_path):
            continue
        with open(img_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        results.append({'filename': fname, 'base64': img_base64, 'score': float(score)})
    return results if results else None

def save_clip_image_features(image_dir: str, features_dir: str):
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        raise ImportError('transformers, torch 패키지가 필요합니다.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    os.makedirs(features_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for fname in image_files:
        img_path = os.path.join(image_dir, fname)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"이미지 열기 실패: {fname} ({e})")
            continue

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs).cpu().numpy()[0]

        # 임베딩 저장
        feature_path = os.path.join(features_dir, fname + ".npy")
        np.save(feature_path, image_features)
        print(f"저장 완료: {feature_path}")

# 직접실행
if __name__ == "__main__":
    image_dir = "./backend/user_photos"
    features_dir = "./backend/features"
    query = "고양이"
    top_n = 5
    similarity_threshold = 0.25  # 원하는 임계값(예: 0.25)로 설정

    results = find_similar_images_by_clip(query, image_dir, features_dir, top_n=top_n)
    if not results:
        print("유사한 이미지가 없습니다.")
    else:
        found = False
        for r in results:
            if r['score'] >= similarity_threshold:
                print(f"파일명: {r['filename']}, 유사도: {r['score']:.4f}")
                img_path = os.path.join(image_dir, r['filename'])
                img = Image.open(img_path)
                img.show()
                found = True
        if not found:
            print(f"유사도 {similarity_threshold} 이상인 이미지는 없습니다.")

    save_clip_image_features(image_dir, features_dir) 