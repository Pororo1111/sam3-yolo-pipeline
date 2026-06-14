

# sam3-yolo-pipeline

YouTube URL · 웹캠 · 로컬 이미지 폴더를 소스로 받아 **SAM3 오토라벨링 → YOLO 학습 → 추론 → 로컬 LLM 영역 설정 & 침입 감지**까지 이어지는 end-to-end 파이프라인 WebUI.

<img width="860" height="530" alt="ezgif com-resize (1)" src="https://github.com/user-attachments/assets/985707be-ce2e-4560-8b88-29e06871db5b" />


## 파이프라인 개요

```
[Tab 1] 소스 선택 & 프레임 추출
    ↓
[Tab 2] SAM3 오토라벨링
    ↓
[Tab 3] 데이터셋 검토 & 구성
    ↓
[Tab 4] YOLO 학습
    ↓
[Tab 5] 추론
    ↓
[Tab 6] 침입 감지 (로컬 LLM 영역 설정 + YOLO 실시간 감시)
```

## 기술 스택

| 역할            | 라이브러리                              |
| --------------- | --------------------------------------- |
| UI              | `gradio`                                |
| YouTube 스트림  | `yt-dlp`                                |
| 영상 처리       | `opencv-python`                         |
| SAM3 오토라벨링 | `ultralytics` — `SAM3SemanticPredictor` |
| YOLO 학습/추론  | `ultralytics` — `YOLO("yolo26n.pt")`    |
| 로컬 LLM (VLM)  | Ollama — `gemma4:e4b` (영역 설정)       |

## 파일 구조

```
yolo-webui/
├── app.py                  ← Gradio 메인 진입점
├── pipeline/
│   ├── extractor.py        ← Tab 1: 프레임 추출
│   ├── labeler.py          ← Tab 2: SAM3 오토라벨링
│   ├── dataset.py          ← Tab 3: 데이터셋 검토/구성
│   ├── trainer.py          ← Tab 4: YOLO 학습
│   ├── inference.py        ← Tab 5: 추론
│   └── zone_monitor.py     ← Tab 6: 침입 감지
├── models/                 ← 사전학습 모델 파일 보관
│   ├── sam3.pt             ← SAM3 세그멘테이션 모델
│   └── yolo26n.pt          ← YOLO 베이스 모델
├── dataset/                ← (gitignore) 추출 프레임 & 라벨
├── runs/                   ← (gitignore) 학습 결과 모델
└── samples/                ← 참고용 Jupyter 노트북
```

## 설치 & 실행

### 1. 환경 설정

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

또는 제공된 스크립트 사용:

```powershell
.\install.sh
```

### 2. 모델 파일 준비

`models/` 폴더에 아래 파일을 배치:

- `sam3.pt` — SAM3 세그멘테이션 모델
- `yolo26n.pt` — YOLO 베이스 모델

### 3. 앱 실행

```powershell
python app.py
```

브라우저에서 `http://localhost:7860` 접속.

## 단계별 사용법

### Tab 1 — 프레임 추출

- YouTube URL · 웹캠 · **이미지 폴더** 중 선택
- 캡처 FPS 설정 후 추출 시작 — 영상 종료/중지 전까지 **무제한 추출**
- 이미지 폴더 모드: 로컬 폴더의 이미지를 그대로 임포트 (한글 경로 지원, "폴더 선택" 버튼으로 탐색)
- 중지 버튼으로 언제든 중단 가능
- 시작 시 기존 추출 프레임은 자동 정리
- 저장 경로: `dataset/raw_frames/`

### Tab 2 — SAM3 오토라벨링

- 텍스트 프롬프트 입력 (예: `person, car, bicycle`)
- 마스크 → YOLO bbox 자동 변환 후 저장
- 라벨링 중 마스크 오버레이 프리뷰 실시간 표시

### Tab 3 — 데이터셋 검토 & 구성

- bbox 오버레이 갤러리로 라벨 품질 확인
- "라벨 없는 프레임 제외" 옵션으로 불량 데이터 필터링
- train/val 비율 설정 후 데이터셋 구성 (재실행 시 기존 train/val을 정리해 데이터 누적·누수 방지)
- `dataset/dataset.yaml` 자동 생성

### Tab 4 — YOLO 학습

- epochs, imgsz, batch, lr0 등 하이퍼파라미터 조정
- 학습 로그 실시간 스트리밍
- 에폭 단위 중지 버튼 지원
- 결과: `runs/detect/train/weights/best.pt`

### Tab 5 — 추론

- YouTube URL · 웹캠 · 이미지 폴더로 추론
- 이미지 폴더 모드: 폴더 내 이미지를 순회하며 추론 (경로 미입력 시 Tab 1에서 고른 폴더 자동 사용)
- 모델 경로 미입력 시 최신 `best.pt` 자동 탐색
- N프레임마다 1회 추론으로 성능 최적화

### Tab 6 — 침입 감지

- YouTube URL · 웹캠 · 이미지 폴더로 스트리밍 (이미지 폴더 경로 미입력 시 Tab 1 폴더 자동 사용)
- 감시 영역을 자연어(한국어 가능)로 입력 → 로컬 LLM(`gemma4:e4b`)이 현재 프레임을 분석해 영역 좌표 자동 설정
- LLM 응답 JSON을 로그 창에 표시
- YOLO로 실시간 객체 탐지 후 zone 내 침입 여부 판별
  - 객체 없음: 초록 테두리
  - 객체 있음: 빨강 테두리 + 반투명 채우기, 내부 객체 수 표시
- 중지 버튼으로 스트림 & 영역 설정 전체 초기화
- Ollama가 `localhost:11434`에서 실행 중이어야 함

## 학습 하이퍼파라미터 가이드

| 파라미터 | 기본값 | 조정 기준                  |
| -------- | ------ | -------------------------- |
| epochs   | 50     | 500장 기준 50~100 적당     |
| imgsz    | 640    | 객체 작으면 416~480 고려   |
| batch    | 16     | VRAM 8GB 이하면 8로 낮추기 |
| lr0      | 0.01   | loss 불안정 시 0.001~0.005 |
| device   | auto   | GPU 자동 감지              |

## 샘플 노트북

`samples/` 폴더에 SAM3 활용 예제 포함:

- `downloader.ipynb` — YouTube 영상 다운로드
- `sam3_point.ipynb` — 포인트 프롬프트 세그멘테이션
- `sam3_anomaly.ipynb` — 이상 탐지
- `sam3_foreign_object.ipynb` — 이물질 탐지
- `sam3_multi_prompt.ipynb` — 다중 프롬프트

## 라이선스

MIT
