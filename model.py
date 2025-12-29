from transformers import AutoModel
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn

class EAT_Classifier(nn.Module):
    def __init__(self, num_classes, lora_r=8, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(
            "worstchan/EAT-base_epoch30_pretrain",
            trust_remote_code=True
        )
        
        config = self.encoder.config
        hidden_size = (getattr(config, "d_model", None) or getattr(config, "embed_dim", None) or getattr(config, "dim", None))
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["qkv", "proj"]
        )
        self.encoder = get_peft_model(self.encoder, lora_config)

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outputs = self.encoder(x)
        pooled = outputs.last_hidden_state[:, 0, :] if hasattr(outputs, "last_hidden_state") else outputs[:, 0, :]
        logits = self.classifier(pooled)
        return logits

    def print_parameters(self):
        """모델의 전체/학습 가능 파라미터 수를 계산하고 출력합니다."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\n--- 모델 파라미터 요약 ---")
        print(f"  - 전체 파라미터: {total_params / 1_000_000:.2f}M")
        print(f"  - 학습 가능 파라미터 (LoRA + Classifier): {trainable_params / 1_000_000:.2f}M")
        print(f"  - 학습 가능 비율: {(trainable_params / total_params) * 100:.2f}%")
        print("--------------------------\n")


if __name__ == '__main__':
    
    num_example_classes = 52 
    model = EAT_Classifier(num_classes=num_example_classes)
    
    # 모델의 파라미터 정보 출력
    model.print_parameters()