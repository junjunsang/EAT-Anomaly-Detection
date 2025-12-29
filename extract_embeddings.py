import torch
from torch.utils.data import DataLoader
from dataset import DCASE_Dataset
from tqdm import tqdm
import os
from model import EAT_Classifier 

def extract_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder_save_path = "best_encoder_model.pth"
    if not os.path.exists(encoder_save_path):
        print(f"'{encoder_save_path}' 파일을 찾을 수 없습니다. 먼저 train.py를 실행하여 모델을 훈련하고 저장해주세요.")
        return

    print("모델 구조를 불러옵니다 (model.py 설정 공유)...")
    temp_model = EAT_Classifier(num_classes=1)
    encoder = temp_model.encoder
    
    print(f"'{encoder_save_path}'에서 학습된 가중치를 로드합니다...")
    encoder.load_state_dict(torch.load(encoder_save_path))
    encoder.to(device)
    encoder.eval()

    print("'train' 폴더의 정상 데이터를 로딩합니다...")
    dataset = DCASE_Dataset("./dev_data", mode='train')
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    all_embeddings = []
    all_filepaths = dataset.file_list

    print("임베딩(CLS 토큰) 추출을 시작합니다...")
    with torch.no_grad():
        for specs, _, _, _ in tqdm(loader, desc="Extracting Embeddings"):
            specs = specs.to(device)
            
            # 인코더 통과
            outputs = encoder(specs)
            
            # CLS 토큰 추출 (시퀀스의 0번째 벡터)
            if hasattr(outputs, "last_hidden_state"):
                # [Batch, Sequence, Dim] -> [Batch, Dim]
                cls_token = outputs.last_hidden_state[:, 0, :]
            else:
                cls_token = outputs[:, 0, :]
            
            # GPU 메모리 절약을 위해 CPU로 이동 후 리스트에 저장
            all_embeddings.append(cls_token.cpu())

    # 리스트를 하나의 텐서로 합치기
    all_embeddings = torch.cat(all_embeddings, dim=0)

    embedding_library = {
        'filepaths': all_filepaths,
        'embeddings': all_embeddings
    }
    
    save_path = "normal_embeddings.pt"
    torch.save(embedding_library, save_path)
    
    print("\n--- 임베딩 추출 및 저장 완료 ---")
    print(f"  - 총 {len(all_filepaths)}개의 임베딩이 추출되었습니다.")
    print(f"  - 임베딩 형태: {all_embeddings.shape}")
    print(f"  - 결과가 '{save_path}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    extract_embeddings()