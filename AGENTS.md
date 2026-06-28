# YOLO 파이프라인 Gradio WebUI

## 응답 언어

- 모든 응답은 항상 한국어로 작성한다.

---

## 프로젝트 개요

YouTube URL · 웹캠 · 비디오 파일 · 로컬 이미지 폴더를 소스로 받아 YOLO 모델을 학습하고 추론하는 end-to-end 파이프라인 WebUI.

## 파이프라인 단계

```
[1] 소스 선택 & 프레임 추출
    ↓
[2] SAM3 오토라벨링
    ↓
[3] 데이터셋 검토 & 구성
    ↓
[4] YOLO 학습
    ↓
[5] 추론
    ↓
[6] 침입 감지 (로컬 LLM 영역 설정 + YOLO 실시간 감시)
```

---

## 레이아웃 (반응형)

- **사이드바 내비게이션** (`gr.Sidebar`) — PC에서는 좌측 고정 사이드바, 모바일에서는 토글 드로어로 자동 전환 (반응형)
- 각 단계는 `gr.Column(visible=...)` 패널로 구성. 사이드바의 `gr.Radio`(`elem_id="pipeline_nav"`) 선택 → `nav.change`가 해당 패널만 표시
- 탭 표시 이름은 번호 접두어("1.", "2.") 제거. 단계 순서는 사이드바 항목 순서(`NAV_STEPS`)로 표현
- 데이터셋 단계 진입 시 `_maybe_load_classes`로 클래스 목록 자동 로드 (기존 `tab3.select` 대체)
- 본문 패널은 `.main-panel`로 최대 폭 제한 + 중앙 정렬

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
├── AGENTS.md
├── app.py                  ← Gradio 메인 진입점
├── install_cuda.sh         ← CUDA PyTorch 설치용 스크립트
├── install_cpu.sh          ← CPU PyTorch 설치용 스크립트
├── pipeline/
│   ├── __init__.py
│   ├── extractor.py        ← Tab 1: 프레임 추출
│   ├── labeler.py          ← Tab 2: SAM3 오토라벨링
│   ├── dataset.py          ← Tab 3: 데이터셋 검토/구성
│   ├── trainer.py          ← Tab 4: YOLO 학습
│   ├── inference.py        ← Tab 5: 추론
│   └── zone_monitor.py     ← Tab 6: 침입 감지
├── dataset/
│   ├── raw_frames/         ← 추출된 원본 프레임
│   ├── images/train|val/   ← 분할된 학습/검증 이미지
│   ├── labels/train|val/   ← YOLO 포맷 라벨
│   └── dataset.yaml        ← 학습용 데이터셋 설정
├── runs/
│   └── detect/train/weights/best.pt  ← 학습 결과 모델
├── samples/                ← 샘플 입력 파일 + 참고용 jupyter notebooks
│   ├── sample_url.txt      ← 샘플 YouTube URL
│   ├── sample.mp4          ← 샘플 비디오
│   ├── sample_image/       ← 샘플 이미지 폴더
│   ├── downloader.ipynb
│   ├── sam3_point.ipynb
│   ├── sam3_anomaly.ipynb
│   ├── sam3_foreign_object.ipynb
│   └── sam3_multi_prompt.ipynb
└── venv/
```

---

## 설치 스크립트

- `install_cuda.sh`: `https://download.pytorch.org/whl/cu126` 인덱스에서 CUDA 빌드 `torch`, `torchvision`을 먼저 설치한다.
- `install_cpu.sh`: `https://download.pytorch.org/whl/cpu` 인덱스에서 CPU 빌드 `torch`, `torchvision`을 먼저 설치한다.
- 두 스크립트 모두 이후 `requirements.txt`에서 `torch`/`torchvision`/`torchaudio` 항목을 제외한 임시 requirements를 만들어 나머지 패키지를 설치한다. PyTorch 빌드가 뒤에서 덮이는 것을 막기 위한 처리다.

---

## 단계별 구현 상세

### Tab 1 — 프레임 추출
**파일**: `pipeline/extractor.py`

- 소스: YouTube URL / 웹캠 / 비디오 파일 / **이미지 폴더**
- YouTube URL, 비디오 파일, 이미지 폴더는 `samples/` 아래 샘플 입력을 카드형 버튼으로 불러올 수 있다.
  - `samples/sample_url.txt`
  - `samples/sample.mp4`
  - `samples/sample_image/`
- 소스별 입력칸과 샘플 버튼 표시는 **클라이언트 JS 토글**(`elem_id` + `.src-hidden` CSS 클래스)로 처리한다. Gradio 큐 왕복 없이 즉시 전환하기 위한 구조다.
- 미리보기는 소스별로 분리한다.
  - YouTube URL: `gr.HTML` iframe embed로 브라우저 네이티브 미리보기 표시
  - 비디오 파일: `gr.HTML` 내부 `<video>` 태그로 브라우저 네이티브 미리보기 표시
  - 웹캠: `gr.Video(sources=["webcam"], streaming=True)`로 클라이언트 측 미리보기 표시
  - 이미지 폴더: `gr.Image(streaming=True)`로 임포트 진행 프레임 표시
- 캡처 저장은 `extractor.capture(..., emit_preview=...)`에서 처리한다. `app.run_capture()`는 이미지 폴더일 때만 `emit_preview=True`로 프레임을 Gradio에 보내고, YouTube/웹캠/비디오 파일은 브라우저 네이티브 미리보기를 쓰면서 상태만 갱신한다.
- YouTube URL은 `yt-dlp`로 스트림 URL을 추출하며 `_STREAM_CACHE_TTL = 600`초 캐시를 둔다.
- YouTube/웹캠/비디오 파일 저장은 `cv2.VideoCapture` 기반으로 읽고, `capture_fps`에 맞춰 `dataset/raw_frames/frame_XXXXX.jpg`에 저장한다.
- 시작 시 `dataset/raw_frames/`의 기존 `frame_*.jpg`를 삭제한 뒤 새로 저장한다.
- 이미지 폴더 모드는 `gr.File(file_count="directory", type="filepath")` 업로드를 사용한다. `_filter_image_paths()`가 지원 확장자만 추려 파일명 기준으로 정렬한다.
- 이미지 폴더 임포트는 jpg/jpeg는 `shutil.copyfile()`로 재인코딩 없이 복사하고, 그 외 포맷은 유니코드 경로 안전 처리를 위해 `np.fromfile` + `cv2.imdecode` 후 jpg로 저장한다.
- 이미지 폴더 미리보기/상태 업데이트는 매 파일마다 보내지 않고 첫 장/마지막/0.5초 간격 중심으로 제한해 대량 복사 시 UI 병목을 줄인다.
- 중지 버튼은 `cancels=[capture_event]`로 캡처 제너레이터를 취소하고 `stop_capture_and_reset()`에서 YouTube/비디오/웹캠/이미지 미리보기를 모두 초기화한다. 상태창은 `extractor.stop()` 반환값으로 직접 갱신한다.

### Tab 2 — SAM3 오토라벨링
**파일**: `pipeline/labeler.py`

- 모델: `sam3.pt` — `SAM3SemanticPredictor` (텍스트 프롬프트 전용)
- 텍스트 프롬프트 입력 → 클래스명 자동 매핑
- 마스크 → YOLO bbox 변환 (x_c y_c w h normalized)
- 저장 경로: `dataset/labels/frame_XXXXX.txt`
- 시작 시 `dataset/labels/`의 평면 `*.txt` 삭제 (이전 소스 오라벨이 새 이미지에 붙는 것 방지). `glob("*.txt")`는 최상위만 매칭 → `train/`·`val/` 하위 폴더는 보존
- `conf=0.25`, `half=False`
- 라벨링 중 마스크 오버레이 프리뷰 실시간 표시 — 진행 미리보기 `label_preview`는 `gr.Image(streaming=True)` (추론 탭과 동일 패턴, 미적용 시 진행 프레임이 갱신 안 됨)
- 미리보기 갤러리(`label_gallery`)는 **CSS로 4열 격자 강제**: Gradio Gallery는 `--grid-cols`를 이미지 수에 맞춰 줄여(1장이면 1열) 이미지가 칸 전체로 커지므로 `#label_gallery .grid-container { grid-template-columns: repeat(4, minmax(0,1fr)) !important }` 로 항상 4열 고정 (`columns=4`만으로는 부족 — 개수가 적으면 열 수가 줄어듦)
- **미리보기 → 전체 실행 2단계 흐름** (유저 편의성):
  - `preview(prompts, conf, n_preview)`: 전체 프레임에서 `np.linspace`로 **균등 샘플링한 N장**만 추론해 갤러리로 표시. **라벨 파일 저장/삭제 없음** → 프롬프트·conf 튜닝 후 결과 확인용
  - `label(prompts, conf)`: 결과가 괜찮으면 전체 프레임에 적용 + 라벨 저장 (기존 동작)
  - 순수 추론부는 `_infer_and_overlay()` 헬퍼로 추출 → preview/label 공유 (rgb 오버레이, label_lines, 객체 수 반환)
  - 중지 버튼은 `cancels=[preview_event, label_event]`로 두 흐름 모두 취소

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

### Tab 3 — 데이터셋 검토 & 구성
**파일**: `pipeline/dataset.py`

- **클래스 편집기 (리스트형)**: 탭 진입/「클래스 불러오기」 시 `scan_classes()`로 라벨을 스캔해 **클래스별 행(ID · 프레임 수 + 즉시 수정 가능한 이름 입력칸)** 을 `@gr.render`로 동적 생성 → 가독성·편의성 향상. 이름 칸은 `gr.Textbox(interactive=True)` (render 동적 생성 시 Gradio 6.x가 입력 추론을 못 해 비활성화되는 것 방지)
  - 이름 우선순위: ① Tab 2 오토라벨링 프롬프트(`label_prompts`, SAM3 프롬프트 순서 = 클래스 ID) → ② 기존 `dataset.yaml` names → ③ `class_{id}`. **최신 라벨링 결과를 항상 우선** 반영 (이전엔 Tab 3 편집값이 우선이라 재라벨링해도 옛 이름이 남던 버그 수정)
  - 이름 편집은 **각 클래스 이름 칸에서 직접** 수행(각 칸은 편집 가능한 `gr.Textbox`) → `.change`로 **즉시** 숨겨진 집계 State(`ds_prompts = gr.State`, ID 0,1,2… 순서 쉼표 결합)에 반영. **별도의 「최종 클래스 이름」 표시 입력칸은 없음** (이전 `interactive=False` Textbox 제거) — `ds_prompts`는 화면에 보이지 않고 `load_preview`/`build_dataset`/`select_frame`/`delete_frame`의 입력 소스로만 사용
  - 이름 수정은 render 트리거 state(`ds_class_state`)를 건드리지 않아 **리렌더/포커스 유실 없음**
  - **편집 보존 vs 자동 갱신**: 탭 진입 시 `_maybe_load_classes`가 직전 로드에 쓴 Tab 2 프롬프트(`ds_last_label` state)와 현재 Tab 2 프롬프트를 비교 → 동일하면(단순 탭 전환) 재스캔 안 함(편집한 이름 보존), 다르면(=Tab 2 재라벨링) 새 프롬프트 이름으로 자동 갱신
  - `_count_class_ids()`는 `labels/*.txt` 최상위만 스캔(`train/`·`val/` 하위 제외). 집계 단위는 **객체(박스) 총합이 아니라 해당 클래스가 들어있는 프레임 수** (한 프레임에 같은 클래스가 여러 개여도 1로 카운트)
- `load_preview(prompts_str, filter_empty)`: raw_frames + labels 매칭 → bbox 오버레이 Gallery 표시
- 통계: 전체 / 라벨 있음 / 라벨 없음 프레임 수
- "라벨 없는 프레임 제외" 체크박스로 불량 데이터 필터링
- **갤러리 이미지 개별 삭제**: 갤러리 클릭 → `select_frame()`이 상세 보기 + 파일명 표시 → 「선택한 이미지 삭제」 → `delete_frame()`이 이미지+라벨 삭제 후 갤러리/통계 갱신
  - 선택은 캡션 파싱이 아니라 **`evt.index` → `_gallery_frames(filter_empty)` 매핑**으로 식별 → Gradio 버전별 `SelectData` 포맷 차이/빈프레임 필터 상태와 무관하게 정확
  - `_gallery_frames()`는 `load_preview` 갤러리와 동일 순서(정렬 + filter_empty 적용)를 보장
- `build_dataset(prompts_str, val_ratio, filter_empty)`: train/val 분할 후 복사
- 분할 전 `images/train|val`, `labels/train|val`을 `shutil.rmtree`로 정리 → 재실행 시 데이터 누적 및 같은 프레임이 train/val 양쪽에 섞이는 누수 방지
- `dataset/dataset.yaml` 자동 생성 (클래스명은 최종 클래스 이름에서 자동 설정)

### Tab 4 — YOLO 학습
**파일**: `pipeline/trainer.py`

- 모델: `yolo26n.pt` (기본 베이스) — **베이스 모델 드롭다운으로 이어학습(파인튜닝) 지원**: 비우면(값 `""`) 사전학습 `yolo26n.pt`로 처음부터, 학습된 모델(`best.pt`)을 고르면 그 가중치 위에 `YOLO(base_path)`로 로딩 후 새 학습. 드롭다운 첫 항목이 "처음부터"이고 이후는 `models.list_trained_models()` 재사용(추론 탭과 동일). 결과는 항상 `name` 입력칸이 정한 새 폴더에 저장돼 베이스는 보존
- stdout 가로채기로 학습 로그 실시간 스트리밍
- ANSI 이스케이프 코드 + `\r` tqdm 패턴 정규식으로 제거 후 표시
- `workers=0` 고정 — Windows DataLoader 멀티프로세싱 spawn 오류 방지
- 에폭 종료 콜백으로 중지 버튼 지원 (`trainer.stop = True`)
- **모델(결과) 이름 입력칸**(`train_name`, 기본 `train`) → `model.train(project=runs/detect, name=<이름>)`. `_safe_name()`이 경로 구분자·금지문자(`<>:"/\|?*`)를 `_`로 치환하고 빈값은 `train`. 같은 이름이 있으면 ultralytics가 숫자를 붙임(덮어쓰지 않음). 이 이름이 추론·침입 감지 탭 모델 드롭다운에 표시됨
- 결과: `runs/detect/<이름>/weights/best.pt` — 실제 저장 폴더는 `on_train_start` 콜백에서 `trainer.save_dir`로 캡처해 정확히 보고(이름 충돌로 숫자가 붙어도 정확)

**하이퍼파라미터** (ultralytics 기본값 기준, 최적화값 아님):

| 파라미터 | 기본값 | 조정 기준 |
|---------|--------|----------|
| epochs | 50 | 500장 기준 50~100 적당. loss 곡선 보고 튜닝 |
| imgsz | 640 | 객체 작거나 해상도 낮으면 416~480 고려 |
| batch | 16 | VRAM 8GB 이하면 8로 낮추기. OOM 시 최우선 조정 |
| lr0 | 0.01 | loss 불안정 시 0.001~0.005로 낮추기 |
| device | auto | GPU 자동 감지. `0` = 첫 번째 GPU 지정 |
| name | train | 결과 폴더명(`runs/detect/<name>`). 추론·침입 감지 모델 목록 표시 이름 |

학습 후 `runs/detect/train/results.csv` 에서 loss 곡선 확인 권장.

### Tab 5 — 추론
**파일**: `pipeline/inference.py`

- 소스: Tab 1과 동일 (YouTube URL / 웹캠 / 비디오 파일 / 이미지 폴더), 영상은 `cv2.VideoCapture` 직접 사용
- 이미지 폴더 모드(`_predict_folder`): 업로드된 이미지를 순회하며 장당 추론·표시, 중지 전까지 반복(0.4초 간격)
- 폴더 입력은 `gr.File(file_count="directory")` 업로드. 업로드가 비어 있으면 **Tab 1 업로드 파일을 자동 상속**(`_inherit_folder`)
- `infer_every`: N프레임마다 1회 추론, 나머지는 마지막 bbox 재사용 (기본값 3)
- `display_interval = 1/15`: 15fps로 yield 제한 → Gradio WebSocket 부담 최소화
- 표시 해상도: 854px 초과 시 리사이즈
- `ultralytics` 로거 ERROR 레벨 설정 — "Waiting for stream" 경고 억제
- **모델 선택은 직접 경로 입력이 아니라 학습된 모델 드롭다운**(`gr.Dropdown`) — `models.list_trained_models()`가 `runs/detect/*/weights/best.pt`를 스캔해 `(표시이름, 경로)`로 반환. 표시이름은 `<run 폴더명>  (YYYY-MM-DD HH:MM 생성일)`, 최신 학습 순 정렬 → 첫 항목 기본 선택. 「모델 목록 새로고침」 버튼 + 탭 진입 시(`panel_inference.select`) 자동 새로고침으로 방금 학습한 모델 즉시 반영
- 드롭다운이 비어(`value=None`) `predict`에 전달돼도 `(model_path or "").strip()`으로 방어 → `_find_best_pt()` 폴백, 그래도 없으면 안내 메시지
- `gr.Image(streaming=True)` 사용

### Tab 6 — 침입 감지
**파일**: `pipeline/zone_monitor.py`

- 소스: Tab 1/5와 동일 (YouTube URL / 웹캠 / 비디오 파일 / 이미지 폴더)
- 이미지 폴더 모드(`_stream_folder`): 업로드 이미지 순회하며 `_last_frame` 갱신(→ 영역 설정 가능) + 침입 판별, 중지 전까지 반복
- 폴더 입력은 `gr.File(file_count="directory")` 업로드. 업로드가 비어 있으면 Tab 1 업로드 파일 자동 상속
- zone 오버레이 로직은 `_render_zones()` 헬퍼로 추출 — 비디오/폴더 루프 공유
- **YOLO 모델 선택은 학습된 모델 드롭다운**(`gr.Dropdown`, `models.list_trained_models()` 공유) — Tab 5 추론과 동일하게 생성 날짜 표시 + 최신순 + 새로고침 버튼/탭 진입 자동 갱신(`panel_zone.select`)
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
