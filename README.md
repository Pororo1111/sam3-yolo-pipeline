# sam3-yolo-pipeline

YouTube URL, 웹캠, 비디오 파일, 이미지 폴더를 입력으로 받아 프레임 추출부터 SAM3 오토라벨링, YOLO 학습, 추론, 침입 감지까지 실행하는 Gradio WebUI입니다.

<img width="860" height="530" alt="app preview" src="https://github.com/user-attachments/assets/985707be-ce2e-4560-8b88-29e06871db5b" />

## 주요 기능

- 프레임 추출: YouTube URL, 웹캠, 비디오 파일, 이미지 폴더 지원
- SAM3 오토라벨링: 텍스트 프롬프트 기반 마스크 생성 및 YOLO bbox 변환
- 데이터셋 검토/구성: 라벨 확인, 클래스명 편집, train/val 분할
- YOLO 학습: 학습 로그 스트리밍, 결과 모델 저장
- 추론: 학습된 모델로 영상/이미지 소스 추론
- 침입 감지: `Safety Cone`을 자동 추적해 라바콘 사이에 감시 영역을 만들고 영역 내 객체 감지
- 샘플 불러오기: `samples/`의 URL, 비디오, 이미지 폴더를 카드 버튼으로 선택

## 설치

CUDA 환경:

```powershell
bash install_cuda.sh
```

CPU 환경:

```powershell
bash install_cpu.sh
```

스크립트는 `venv`를 만들거나 재사용한 뒤 PyTorch 빌드(CUDA/CPU)를 먼저 설치하고, 나머지 의존성을 설치합니다.

## 모델 준비

`models/` 폴더에 아래 파일을 배치합니다.

- `sam3.pt` — Hugging Face에서 모델 접근 승인을 받은 뒤 다운로드: https://huggingface.co/facebook/sam3
- `yolo26n.pt` — 없으면 Ultralytics가 필요 시 자동 다운로드

## 실행

```powershell
venv\Scripts\Activate.ps1
gradio app.py
```

브라우저에서 `http://localhost:7860`에 접속합니다.

웹캠 목록은 **앱 서버에 연결된 카메라**를 먼저 표시하고, `서버 + 접속 기기
카메라 검색`을 누르면 카메라 권한을 요청한 뒤 현재 브라우저(모바일 포함)의
카메라를 이어서 표시합니다. 접속 기기 카메라를 선택한 경우 표시되는 카메라
입력에서 스트리밍을 먼저 시작한 뒤 프레임 추출·추론·침입 감지를 시작하세요.

브라우저 카메라 API는 보안 컨텍스트에서만 동작하므로 원격 모바일 접속에는
**HTTPS가 필요**합니다(`localhost` 접속만 HTTP 예외). Raspberry Pi의 서버
카메라는 USB/V4L2 장치(`/dev/video*`) 권한과 다른 프로세스의 카메라 점유
여부를 먼저 확인하세요.

## 폴더 구조

```text
yolo-webui/
├── app.py
├── install_cuda.sh
├── install_cpu.sh
├── pipeline/
│   ├── extractor.py
│   ├── labeler.py
│   ├── dataset.py
│   ├── dataset_importer.py
│   ├── trainer.py
│   ├── inference.py
│   └── zone_monitor.py
├── samples/
│   ├── sample_url.txt
│   ├── sample.mp4
│   └── sample_image/
├── dataset/   # 추출 프레임, 라벨, 구성된 학습 데이터
├── models/    # 모델 가중치
└── runs/      # 학습 결과
```

## 라이선스

MIT
