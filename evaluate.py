import torch
from torch.utils.data import DataLoader
from dataset import DCASE_Dataset
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import argparse

# 👇 여기부터 model.py를 가져오는 방식으로 변경!
from model import EAT_Classifier

def evaluate_per_machine_type(k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder_save_path = "best_encoder_model.pth"
    embedding_library_path = "normal_embeddings.pt"

    if not os.path.exists(encoder_save_path) or not os.path.exists(embedding_library_path):
        print(f"필요한 파일이 없습니다.")
        return
        
    print("모델 구조를 로딩합니다 (model.py 설정 공유)...")
    
    # 1. 껍데기 모델 생성 (평가만 할 거라 num_classes는 아무 숫자나 OK)
    # model.py에 있는 LoRA 설정(r=8, alpha=32 등)을 그대로 가져옵니다.
    temp_model = EAT_Classifier(num_classes=1)
    
    # 2. 필요한 'encoder' 부분만 분리
    encoder = temp_model.encoder

    print(f"'{encoder_save_path}'에서 학습된 가중치를 로드합니다...")
    encoder.load_state_dict(torch.load(encoder_save_path))
    encoder.to(device)
    encoder.eval()

    library = torch.load(embedding_library_path)
    normal_embeddings = library['embeddings'].to(device)

    print(f"'test' 데이터셋의 이상 점수를 K={k}인 KNN 방식으로 계산합니다...")
    test_dataset = DCASE_Dataset("./dev_data", mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    ground_truth_labels = []
    anomaly_scores = []
    machine_types = []

    with torch.no_grad():
        for spec, _, is_normal_batch, machine_type_str_batch in tqdm(test_loader, desc="Evaluating"):
            spec = spec.to(device)
            is_normal = is_normal_batch[0].item()
            machine_type = machine_type_str_batch[0]
            
            # 인코더 통과
            outputs = encoder(spec)
            
            # CLS 토큰 추출 (시퀀스의 0번째 벡터)
            if hasattr(outputs, "last_hidden_state"):
                test_embedding = outputs.last_hidden_state[:, 0, :]
            else:
                test_embedding = outputs[:, 0, :]
            
            # 코사인 유사도를 이용한 거리 계산 (1 - 유사도 = 거리)
            # test_embedding: (1, dim), normal_embeddings: (N, dim) -> 브로드캐스팅 됨
            distances = 1 - F.cosine_similarity(test_embedding, normal_embeddings)
            
            # 가장 가까운 K개의 이웃 찾기
            top_k_distances, _ = torch.topk(distances, k, largest=False)
            
            # 평균 거리를 이상 점수(Anomaly Score)로 사용
            score = torch.mean(top_k_distances).item()
            
            anomaly_scores.append(score)
            ground_truth_labels.append(0 if is_normal else 1) # 0: normal, 1: anomaly
            machine_types.append(machine_type)

    print("\n--- 분석 완료 ---")
    
    # 전체 성능
    overall_auroc = roc_auc_score(ground_truth_labels, anomaly_scores)
    print(f"전체 평균 성능 (Overall AUROC): {overall_auroc:.4f}\n")
    
    # 타입별 성능
    print("--- 기계 타입별 성능 ---")
    unique_machine_types = sorted(list(set(machine_types)))
    scores = np.array(anomaly_scores)
    labels = np.array(ground_truth_labels)
    types = np.array(machine_types)
    
    for machine_type in unique_machine_types:
        mask = (types == machine_type)
        type_scores = scores[mask]
        type_labels = labels[mask]
        
        if len(np.unique(type_labels)) > 1:
            type_auroc = roc_auc_score(type_labels, type_scores)
            print(f"  - {machine_type.ljust(10)}: {type_auroc:.4f}")
        else:
            print(f"  - {machine_type.ljust(10)}: (비정상 샘플 없음)")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN 기반 이상 탐지 평가 스크립트 (타입별 분석 포함)")
    parser.add_argument("-k", type=int, default=1, help="이상 점수 계산에 사용할 이웃(K)의 수")
    args = parser.parse_args()
    
    evaluate_per_machine_type(args.k)