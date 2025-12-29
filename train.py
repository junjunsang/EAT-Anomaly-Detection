import torch
from torch.utils.data import DataLoader
from dataset import DCASE_Dataset
from model import EAT_Classifier
from tqdm import tqdm
import multiprocessing
import numpy as np
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DCASE_Dataset("./dev_data", mode='train')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    
    test_dataset = DCASE_Dataset("./dev_data", mode='test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    num_total_classes = train_dataset.num_classes
    print(f"--- 총 {num_total_classes}개의 조합 라벨로 분류를 시작합니다 ---")
    model = EAT_Classifier(num_classes=num_total_classes).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    save_path = "best_encoder_model.pth"


    for epoch in range(1):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", ncols=120)
        
      
        for specs, labels, _, _ in pbar_train:
        # ------------------------------------
            specs, labels = specs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            pbar_train.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Acc": f"{(total_correct/total_samples)*100:.2f}%"
            })

        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        with torch.no_grad():

            for specs, labels, _, _ in test_loader:
                specs, labels = specs.to(device), labels.to(device)
                logits = model(specs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = (total_correct / total_samples) * 100
        avg_val_loss = val_loss / len(test_loader)
        val_acc = (val_correct / val_samples) * 100

        print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.encoder.state_dict(), save_path)
            print(f"  -> 최고 검증 정확도 경신! ({best_val_acc:.2f}%) 모델을 '{save_path}'에 저장했습니다.")
    
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()