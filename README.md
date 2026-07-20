# sam3-yolo-pipeline

YouTube URL, 웹캠, 비디오 파일, 이미지 폴더를 입력으로 받아 프레임 추출부터 SAM3 오토라벨링, YOLO 학습, 추론, 침입 감지까지 실행하는 Gradio WebUI입니다.

<img width="860" height="530" alt="app preview" src="https://github.com/user-attachments/assets/985707be-ce2e-4560-8b88-29e06871db5b" />

## 주요 기능

- 프레임 추출: YouTube URL, 웹캠, 비디오 파일, 이미지 폴더 지원
- SAM3 오토라벨링: 텍스트 프롬프트 기반 마스크 생성 및 YOLO bbox 변환
- 데이터셋 검토/구성: 라벨 확인, 클래스명 편집, train/val 분할
- 외부 데이터셋 불러오기: 여러 YOLO ZIP을 검사·등록하고 공동 학습
- YOLO 학습: 학습 로그 스트리밍, 결과 모델 저장
- 추론: 학습된 모델로 영상/이미지 소스 추론
- 침입 감지: 로컬 LLM으로 감시 영역을 설정하고 YOLO로 영역 내 객체 감지
- 수동/추적 영역: 고정 프레임을 마우스로 클릭해 다각형을 만들거나, 클릭한 라바콘을 ByteTrack으로 추적
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
python app.py
```

브라우저에서 `http://localhost:7860`에 접속합니다.

웹캠은 브라우저가 열린 PC의 카메라가 아니라 **앱 서버에 연결된 카메라**를
OpenCV로 사용합니다. Raspberry Pi에서는 USB/V4L2 장치(`/dev/video*`) 권한과
다른 프로세스의 카메라 점유 여부를 먼저 확인하세요.

## 외부 YOLO 데이터셋 불러오기

**데이터셋 불러오기** 탭에서 폴더 구조를 보존한 YOLO 데이터셋 ZIP을 하나
이상 선택한 뒤 **ZIP 압축 해제 후 등록**을 누릅니다.

등록 시 train/val 경로, 클래스, 이미지 수, 라벨 형식과 bbox 좌표를 검사합니다.
**학습에 사용할 데이터셋**에서 여러 항목을 체크하면 학습 직전에 다시 검사한 뒤
하나의 조합 데이터셋으로 학습합니다. 클래스 구성이 같으면 원본 이미지 경로를
직접 결합합니다. 클래스 이름이나 ID 순서가 다르면 통합 클래스 목록을 만들고,
관리형 학습 디렉터리에서 라벨 ID만 자동 재매핑합니다. 예를 들어 `person`과
`Safety Cone` 데이터셋을 함께 선택하면 각각 ID 0과 1인 2개 클래스 모델을
학습합니다. 원본 YAML과 라벨은 변경하지 않습니다.

ZIP 데이터셋은 안전 검사를 거쳐 `dataset/external/`에 압축 해제되며, 조합 YAML은
`dataset/.combined/`에 생성됩니다. 클래스 재매핑이 필요할 때 이미지는 가능한
경우 hardlink로 연결하고, 다른 디스크처럼 hardlink가 불가능한 파일만 복사합니다.
첫 재매핑 때는 staged 라벨을 생성하므로 데이터셋 크기에 따라 시간이 걸릴 수
있지만, 원본 경로·클래스·라벨이 바뀌지 않으면 검증된 캐시를 재사용합니다.
`dataset/imported/` 경로에는 의존하지 않습니다.

## 네트워크에서 서버 열기

다른 기기에서 WebUI에 접속하려면 서버 주소와 UI 인증 정보를 설정합니다.

```powershell
$env:YOLO_APP_HOST="0.0.0.0"
$env:YOLO_UI_USER="admin"
$env:YOLO_UI_PASSWORD="충분히-긴-암호"
python app.py
```

외부 주소(`0.0.0.0` 포함)에 bind할 때는 UI 인증이 필수입니다. 인증을 담당하는
HTTPS 역방향 프록시 뒤에서만 `YOLO_ALLOW_UNAUTHENTICATED_UI=1`을 사용하세요.

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

## 외부 서비스

침입 감지의 자연어 영역 설정을 쓰려면 Ollama가 실행 중이어야 합니다.

```powershell
ollama serve
```

사용 모델: `gemma4:e4b`. 침입 감지 화면의 **Gemma 모델 다운로드** 버튼이
Ollama `/api/pull` 진행률을 표시하며 모델을 설치합니다. CLI에서는 다음과 같이
설치할 수 있습니다.

```powershell
ollama pull gemma4:e4b
```

Ollama 주소는 `OLLAMA_BASE_URL`로 바꿀 수 있으며 기본값은
`http://localhost:11434`입니다.

### 수동 다각형과 라바콘 추적 영역

1. 침입 감지 스트림을 시작합니다.
2. **현재 프레임 가져오기**를 눌러 편집 프레임을 고정합니다.
3. `수동 다각형`에서는 꼭짓점을 순서대로 클릭합니다.
4. `라바콘 추적 (ByteTrack)`에서는 같은 클래스의 라바콘 bbox를 3개 이상 순서대로 클릭합니다.
5. **다각형 완료**를 누릅니다. 추적 영역이 있으면 앱이 추론 간격을 자동으로 1로 사용합니다.

라바콘 꼭짓점은 bbox 바닥 중앙과 ByteTrack ID를 따라 움직입니다. 잠시 가려져
Track이 유실되면 마지막 좌표를 유지하고 영역을 주황색으로 표시합니다. 영역
경계로 선택한 라바콘은 침입 객체 수에서 제외됩니다.

## 라이선스

MIT
