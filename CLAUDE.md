# YOLO 파이프라인 Gradio WebUI

## 프로젝트 개요

YouTube URL 또는 웹캠을 소스로 받아 YOLO 모델을 학습하고 추론하는 end-to-end 파이프라인 WebUI.

## 파이프라인 단계

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

---

## 기술 스택

| 역할 | 라이브러리 |
|------|-----------|
| UI | `gradio` |
| YouTube 스트림 | `yt-dlp` |
| 영상 처리 | `opencv-python` |
| SAM3 오토라벨링 | `ultralytics` — `SAM3SemanticPredictor` |
| YOLO 학습/추론 | `ultralytics` — `YOLO("yolo26n.pt")` |
| 로컬 LLM (VLM) | `ollama` REST API — `gemma4:e4b` (영역 설정) |
| HTTP 클라이언트 | `requests` — Ollama API 호출 |

---

## 파일 구조

```
yolo-webui/
├── CLAUDE.md
├── app.py                  ← Gradio 메인 진입점
├── pipeline/
│   ├── __init__.py
│   ├── extractor.py        ← Tab 1: 프레임 추출         (완료)
│   ├── labeler.py          ← Tab 2: SAM3 오토라벨링     (완료)
│   ├── dataset.py          ← Tab 3: 데이터셋 검토/구성  (완료)
│   ├── trainer.py          ← Tab 4: YOLO 학습           (완료)
│   ├── inference.py        ← Tab 5: 추론                (완료)
│   └── zone_monitor.py     ← Tab 6: 침입 감지           (완료)
├── dataset/
│   ├── raw_frames/         ← 추출된 원본 프레임
│   ├── images/train|val/   ← 분할된 학습/검증 이미지
│   ├── labels/train|val/   ← YOLO 포맷 라벨
│   └── dataset.yaml        ← 학습용 데이터셋 설정
├── runs/
│   └── detect/train/weights/best.pt  ← 학습 결과 모델
├── samples/                ← 참고용 jupyter notebooks
│   ├── downloader.ipynb
│   ├── sam3_point.ipynb
│   ├── sam3_anomaly.ipynb
│   ├── sam3_foreign_object.ipynb
│   └── sam3_multi_prompt.ipynb
└── venv/
```

---

## 단계별 구현 상세

### Tab 1 — 프레임 추출 (완료)
**파일**: `pipeline/extractor.py`

- 소스: YouTube URL (`yt-dlp` 스트림 URL 추출) 또는 웹캠 (`cv2.VideoCapture(0)`)
- 고정 간격 캡처: `frame_interval = src_fps / capture_fps`
- 목표: 500장, 중지 버튼으로 언제든 중단 가능
- 저장 경로: `dataset/raw_frames/frame_XXXXX.jpg`
- Gradio 중지: `_stop_event`(threading.Event) + `cancels=[capture_event]` 동시 사용
- 프리뷰: capture_fps 간격으로만 yield → Gradio WebSocket 부담 최소화

### Tab 2 — SAM3 오토라벨링 (완료)
**파일**: `pipeline/labeler.py`

- 모델: `sam3.pt` — `SAM3SemanticPredictor` (텍스트 프롬프트 전용)
- 텍스트 프롬프트 입력 → 클래스명 자동 매핑
- 마스크 → YOLO bbox 변환 (x_c y_c w h normalized)
- 저장 경로: `dataset/labels/frame_XXXXX.txt`
- `conf=0.25`, `half=False`
- 라벨링 중 마스크 오버레이 프리뷰 실시간 표시

```python
# SAM3 초기화 패턴
from ultralytics.models.sam import SAM3SemanticPredictor
overrides = dict(conf=0.25, task="segment", mode="predict",
                 model="sam3.pt", half=False, save=False)
predictor = SAM3SemanticPredictor(overrides=overrides)

# 추론 패턴
predictor.set_image(frame)
results = predictor(text=["person", "car"])
masks = results[0].masks.data.cpu().numpy().astype(np.uint8)
cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
```

### Tab 3 — 데이터셋 검토 & 구성 (완료)
**파일**: `pipeline/dataset.py`

- `load_preview(prompts_str, filter_empty)`: raw_frames + labels 매칭 → bbox 오버레이 Gallery 표시
- 통계: 전체 / 라벨 있음 / 라벨 없음 프레임 수
- "라벨 없는 프레임 제외" 체크박스로 불량 데이터 필터링
- `build_dataset(prompts_str, val_ratio, filter_empty)`: train/val 분할 후 복사
- `dataset/dataset.yaml` 자동 생성 (클래스명은 프롬프트에서 자동 설정)

### Tab 4 — YOLO 학습 (완료)
**파일**: `pipeline/trainer.py`

- 모델: `yolo26n.pt`
- stdout 가로채기로 학습 로그 실시간 스트리밍
- ANSI 이스케이프 코드 + `\r` tqdm 패턴 정규식으로 제거 후 표시
- `workers=0` 고정 — Windows DataLoader 멀티프로세싱 spawn 오류 방지
- 에폭 종료 콜백으로 중지 버튼 지원 (`trainer.stop = True`)
- 결과: `runs/detect/train/weights/best.pt`

**하이퍼파라미터** (ultralytics 기본값 기준, 최적화값 아님):

| 파라미터 | 기본값 | 조정 기준 |
|---------|--------|----------|
| epochs | 50 | 500장 기준 50~100 적당. loss 곡선 보고 튜닝 |
| imgsz | 640 | 객체 작거나 해상도 낮으면 416~480 고려 |
| batch | 16 | VRAM 8GB 이하면 8로 낮추기. OOM 시 최우선 조정 |
| lr0 | 0.01 | loss 불안정 시 0.001~0.005로 낮추기 |
| device | auto | GPU 자동 감지. `0` = 첫 번째 GPU 지정 |

학습 후 `runs/detect/train/results.csv` 에서 loss 곡선 확인 권장.

### Tab 5 — 추론 (완료)
**파일**: `pipeline/inference.py`

- 소스: Tab 1과 동일 (YouTube URL or 웹캠), `cv2.VideoCapture` 직접 사용
- `infer_every`: N프레임마다 1회 추론, 나머지는 마지막 bbox 재사용 (기본값 3)
- `display_interval = 1/15`: 15fps로 yield 제한 → Gradio WebSocket 부담 최소화
- 표시 해상도: 854px 초과 시 리사이즈
- `ultralytics` 로거 ERROR 레벨 설정 — "Waiting for stream" 경고 억제
- 모델 경로 미입력 시 `runs/detect/` 에서 최신 `best.pt` 자동 탐색
- `gr.Image(streaming=True)` 사용

### Tab 6 — 침입 감지 (완료)
**파일**: `pipeline/zone_monitor.py`

- 소스: Tab 1/5와 동일 (YouTube URL or 웹캠)
- YOLO 모델 경로 미입력 시 `runs/detect/`에서 최신 `best.pt` 자동 탐색
- **영역 설정 흐름**:
  1. 스트림 시작 → 마지막 프레임을 `_last_frame`에 계속 저장
  2. 사용자가 감시 영역을 한국어로 입력
  3. "영역 설정" 클릭 → 현재 프레임 + 프롬프트를 Ollama(`gemma4:e4b`)에 전송
  4. LLM이 정규화 좌표 JSON 반환 → `_zones`에 저장
  5. LLM 응답 원문은 로그 창에 표시
- **실시간 침입 판별**:
  - `cv2.pointPolygonTest`로 bbox 중심점이 zone 내부인지 검사
  - zone 내 객체 없음: 초록 테두리
  - zone 내 객체 있음: 빨강 테두리 + 반투명 빨강 채우기(25%)
  - zone 라벨에 `(N)` 형태로 내부 객체 수 실시간 표시
- `infer_every`: N프레임마다 1회 추론 (기본값 3)
- `display_interval = 1/15`: 15fps yield 제한
- zone 라벨은 항상 영어로 출력 (시스템 프롬프트 강제)
- Ollama API: `http://localhost:11434/api/chat`, `format="json"` 강제
- 중지 버튼: 스트림 종료 + `_zones` 초기화 + `_last_frame` 초기화

---

## SAM3 주의사항

- `SAM3SemanticPredictor` — 텍스트 프롬프트 전용
- `SAM("sam3.pt")` — 포인트/박스 시각 프롬프트 전용 (두 클래스 혼용 불가)
- mask dtype이 `bool`인 경우 cv2.resize 에러 발생 → `.astype(np.uint8)` 필수
- `imgsz` 는 stride 14의 배수여야 함 (경고 발생 시 자동 보정됨)

---

## 실행

```powershell
# venv 활성화
c:\Users\moula\source\yolo-webui\venv\Scripts\Activate.ps1

# 앱 실행
python app.py
```
