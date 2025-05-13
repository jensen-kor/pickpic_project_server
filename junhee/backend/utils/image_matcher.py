import os
import torch
from PIL import Image
import base64
import numpy as np
from typing import Optional, List

def find_similar_images_by_clip(text: str, image_dir: str, features_dir: str, top_n: int = 5, similarity_threshold: float = 0.0) -> Optional[List[dict]]:
    """
    띄어쓰기로 구분된 여러 키워드가 들어오면 각 키워드별로 영어로 번역 후 따로 검색해서
    모든 키워드에 해당하는 이미지를 top_n개씩 합쳐서 반환 (중복 제거, 유사도는 최대값)
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        raise ImportError('transformers, torch 패키지가 필요합니다.')
    try:
        from googletrans import Translator
    except ImportError:
        raise ImportError('googletrans 패키지가 필요합니다. (pip install googletrans==4.0.0-rc1)')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14', use_fast=False)

    # 이미지 임베딩(.npy) 파일 목록
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
    if not feature_files:
        return None

    # 여러 키워드로 분리 및 번역
    keywords = text.strip().split()
    translator = Translator()
    translated_keywords = []
    for kw in keywords:
        try:
            translated = translator.translate(kw, src='ko', dest='en').text
            translated_keywords.append(translated)
        except Exception:
            translated_keywords.append(kw)  # 번역 실패시 원본 사용

    image_scores = dict()  # {파일명: (유사도, 번역된 키워드)}

    for idx, keyword in enumerate(translated_keywords):
        # 텍스트 임베딩 추출
        inputs = processor(text=[keyword], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs).cpu().numpy()[0]

        similarities = []
        for feat_file in feature_files:
            img_feature = np.load(os.path.join(features_dir, feat_file))
            sim = np.dot(text_features, img_feature) / (np.linalg.norm(text_features) * np.linalg.norm(img_feature))
            similarities.append((feat_file.replace('.npy', ''), sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        for fname, score in similarities[:top_n]:
            if score < similarity_threshold:
                continue
            # 여러 키워드에 걸릴 경우 더 높은 유사도로 갱신
            if fname not in image_scores or image_scores[fname][0] < score:
                image_scores[fname] = (score, keywords[idx], keyword)  # (유사도, 원본, 번역)

    # 유사도 내림차순 정렬
    sorted_images = sorted(image_scores.items(), key=lambda x: x[1][0], reverse=True)
    results = []
    for fname, (score, orig_keyword, trans_keyword) in sorted_images:
        img_path = os.path.join(image_dir, fname)
        if not os.path.exists(img_path):
            continue
        with open(img_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        results.append({'filename': fname, 'base64': img_base64, 'score': float(score), 'matched_keyword': orig_keyword, 'translated_keyword': trans_keyword})
    return results if results else None

def save_clip_image_features(image_dir: str, features_dir: str):
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        raise ImportError('transformers, torch 패키지가 필요합니다.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14', use_fast=False)

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
    query = "원숭이"
    top_n = 5
    similarity_threshold = 0.2

    results = find_similar_images_by_clip(query, image_dir, features_dir, top_n=top_n, similarity_threshold=similarity_threshold)
    if not results:
        print("유사한 이미지가 없습니다.")
    else:
        for r in results:
            print(f"파일명: {r['filename']}, 유사도: {r['score']:.4f}, 매칭 키워드: {r['matched_keyword']}, 번역된 키워드: {r['translated_keyword']}")
            img_path = os.path.join(image_dir, r['filename'])
            img = Image.open(img_path)
            img.show()

    save_clip_image_features(image_dir, features_dir) 