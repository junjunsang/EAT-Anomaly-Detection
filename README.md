#  DCASE Anomaly Detection with EAT + LoRA

이 프로젝트는 DCASE Task 2(기계 이상음 탐지)를 위한 딥러닝 모델입니다. 사전 학습된 **EAT(Efficient Audio Transformer)** 모델에 **LoRA**를 적용하여 효율적으로 학습하고, 정상 데이터의 임베딩을 저장하여 **KNN 기반의 거리 계산**을 통해 비정상 여부를 판별합니다.

##  Key Features

* **Model Architecture**: `worstchan/EAT-base_epoch30_pretrain` 기반.
* 
**Efficient Fine-tuning**: **LoRA (Low-Rank Adaptation)** 를 적용하여 적은 파라미터로 효과적인 전이 학습(Transfer Learning) 수행.


* **Anomaly Detection**: 정상 데이터의 임베딩을 Memory Bank로 구축하고, 테스트 데이터와의 Cosine Distance를 기반으로 KNN Scoring 수행.
* **Preprocessing**: Log-Mel Spectrogram 변환.

##  Installation

이 프로젝트는 **`uv`** 패키지 매니저를 사용하여 의존성을 관리합니다.

```bash
# 1. 저장소 클론
git clone https://github.com/junjunsang/EAT-Anomaly-Detection.git
cd EAT-Anomaly-Detection

# 2. 의존성 설치 및 가상환경 세팅 (CUDA 12.1 PyTorch 자동 설치)
uv sync

```

##  Data Preparation

데이터셋은 프로젝트 루트 경로의 `dev_data` 폴더에 위치해야 합니다.
DCASE 데이터셋 구조를 따르며, 학습용(`train`)과 테스트용(`test`)으로 구분됩니다.

```plaintext
dcase-anomaly-detection/
├── dev_data/
│   ├── train/
│   │   ├── fan/
│   │   ├── valve/
│   │   └── ...
│   └── test/
│       ├── fan/
│       ├── valve/
│       └── ...

```

##  Usage

이상 탐지 프로세스는 **학습(Train) → 추출(Extract) → 평가(Evaluate)** 의 3단계로 진행됩니다.

### 1. Model Training (Fine-tuning)

LoRA를 적용한 인코더를 학습시켜 기계음의 특징을 추출하도록 만듭니다. 학습이 완료되면 `best_encoder_model.pth`가 저장됩니다.

```bash
uv run train.py

```

### 2. Embedding Extraction

학습된 인코더를 사용하여 정상(Normal) 데이터의 임베딩 벡터를 추출하고, 이를 `normal_embeddings.pt` 라이브러리 파일로 저장합니다.

```bash
uv run extract_embeddings.py

```

### 3. Anomaly Evaluation

저장된 정상 임베딩 라이브러리와 테스트 데이터 간의 **KNN 거리(Cosine Similarity)** 를 계산하여 이상 점수(Anomaly Score)를 측정하고 AUROC 성능을 출력합니다.

```bash
# 기본 실행 (K=1)
uv run evaluate.py

# K값 변경 실행 (예: K=5)
uv run evaluate.py -k 5

```

##  Project Structure

```plaintext
├── dev_data/              # 데이터셋 폴더 (gitignore 처리됨)
├── dataset.py             # DCASE 데이터셋 로더 및 라벨링 처리
├── preprocessing.py       # Mel Spectrogram 변환 로직
├── model.py               # EAT 모델 및 LoRA 설정 정의
├── train.py               # 분류기 학습 및 인코더 가중치 저장
├── extract_embeddings.py  # 정상 데이터 임베딩 추출 및 저장
├── evaluate.py            # KNN 기반 이상 탐지 성능 평가 (AUROC)
├── pyproject.toml         # 프로젝트 의존성 및 설정 관리
└── README.md              # 프로젝트 설명 문서

```

##  Performance

* **Metric**: AUROC (Area Under the ROC Curve)
* **Method**: KNN Anomaly Scoring (Cosine Distance)
* 평가 스크립트 실행 시 전체 평균 성능(Overall AUROC)과 기계 타입(Machine Type)별 성능을 확인할 수 있습니다.

