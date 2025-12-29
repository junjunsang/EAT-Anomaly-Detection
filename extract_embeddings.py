import torch
from torch.utils.data import DataLoader
from dataset import DCASE_Dataset
from tqdm import tqdm
import os

from transformers import AutoModel
from peft import LoraConfig, get_peft_model

def extract_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("인코더 구조를 생성합니다...")
    encoder = AutoModel.from_pretrained(
        "worstchan/EAT-base_epoch30_pretrain",
        trust_remote_code=True
    )
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["qkv", "proj"]
    )
    # ----------------------------------------------------
    
    encoder = get_peft_model(encoder, lora_config)

    encoder_save_path = "best_encoder_model.pth"
    if not os.path.exists(encoder_save_path):
        print(f"'{encoder_save_path}' 파일을 찾을 수 없습니다. 먼저 train.py를 실행하여 모델을 훈련하고 저장해주세요.")
        return
        
    print(f"'{encoder_save_path}'에서 훈련된 인코더 가중치를 불러옵니다...")
    encoder.load_state_dict(torch.load(encoder_save_path))
    encoder.to(device)
    encoder.eval()

    print("'train' 폴더의 정상 데이터를 로딩합니다...")
    dataset = DCASE_Dataset("./dev_data", mode='train')
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    all_embeddings = []
    all_filepaths = dataset.file_list

    print("임베딩 추출을 시작합니다...")
    with torch.no_grad():
    
        for specs, _, _, _ in tqdm(loader, desc="Extracting Embeddings"):
            specs = specs.to(device)
            
            outputs = encoder(specs)
            
            if hasattr(outputs, "last_hidden_state"):
                pooled_output = outputs.last_hidden_state[:, 0, :]
            else:
                pooled_output = outputs[:, 0, :]
            
            all_embeddings.append(pooled_output.cpu())

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