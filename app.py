import html as _html
from pathlib import Path

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from pipeline import (
    dataset,
    dataset_importer,
    extractor,
    inference,
    labeler,
    models,
    trainer,
    webcams,
    zone_monitor,
)


load_dotenv()


SAMPLES_DIR = Path("samples")
SAMPLE_URL_PATH = SAMPLES_DIR / "sample_url.txt"
SAMPLE_VIDEO_PATH = SAMPLES_DIR / "sample.mp4"
SAMPLE_IMAGE_DIR = SAMPLES_DIR / "sample_image"
SAMPLE_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def _inherit_folder(source_type, current_files, tab1_files):
    """이미지 폴더 선택 시, 업로드가 비어 있으면 Tab 1의 업로드 파일을 물려받는다."""
    if source_type == "이미지 폴더" and not current_files:
        return gr.update(value=tab1_files or None)
    return gr.update()


def _inherit_video(source_type, current_file, tab1_file):
    """비디오 파일 선택 시, 업로드가 비어 있으면 Tab 1의 업로드 파일을 물려받는다."""
    if source_type == "비디오 파일" and not current_file:
        return gr.update(value=tab1_file or None)
    return gr.update()


def load_sample_youtube_url():
    """samples/sample_url.txt의 URL을 YouTube 입력칸에 채운다."""
    if not SAMPLE_URL_PATH.exists():
        return gr.update(), f"샘플 URL 파일을 찾을 수 없습니다: {SAMPLE_URL_PATH}"
    url = SAMPLE_URL_PATH.read_text(encoding="utf-8", errors="ignore").strip()
    if not url:
        return gr.update(), f"샘플 URL 파일이 비어 있습니다: {SAMPLE_URL_PATH}"
    return gr.update(value=url), "샘플 YouTube URL을 불러왔습니다."


def load_sample_video_file():
    """samples/sample.mp4를 비디오 파일 입력에 채운다."""
    if not SAMPLE_VIDEO_PATH.exists():
        return gr.update(), f"샘플 비디오 파일을 찾을 수 없습니다: {SAMPLE_VIDEO_PATH}"
    return gr.update(value=str(SAMPLE_VIDEO_PATH)), "샘플 비디오 파일을 선택했습니다."


def load_sample_image_folder():
    """samples/sample_image 폴더의 이미지들을 이미지 폴더 입력에 채운다."""
    if not SAMPLE_IMAGE_DIR.exists():
        return gr.update(), f"샘플 이미지 폴더를 찾을 수 없습니다: {SAMPLE_IMAGE_DIR}"
    files = sorted(
        str(path)
        for path in SAMPLE_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SAMPLE_IMAGE_EXTS
    )
    if not files:
        return gr.update(), f"샘플 이미지 폴더에 이미지 파일이 없습니다: {SAMPLE_IMAGE_DIR}"
    return gr.update(value=files), f"샘플 이미지 폴더를 선택했습니다. ({len(files)}장)"


def run_capture(
    browser_session_id,
    source_type,
    youtube_url,
    capture_fps,
    folder_files,
    webcam_index,
    video_file,
):
    yield from extractor.capture(
        source_type,
        youtube_url,
        int(capture_fps),
        folder_files,
        webcam_index,
        video_file,
        browser_session_id,
    )


def prepare_capture_preview():
    """공용 이미지 스트림을 비우고 새 캡처를 준비한다."""
    return None, "실시간 미리보기를 준비합니다."


def stop_capture_and_reset():
    """캡처 중지 후 미리보기 영역을 초기 상태로 되돌린다."""
    status = extractor.stop()
    return None, status


def run_inference(
    browser_session_id,
    model_path,
    source_type,
    youtube_url,
    conf,
    infer_every,
    folder_files,
    webcam_index,
    video_file,
):
    yield from inference.predict(
        model_path,
        source_type,
        youtube_url,
        conf,
        infer_every,
        folder_files,
        webcam_index,
        video_file,
        browser_session_id,
    )


def run_zone_stream(
    browser_session_id,
    session_id,
    source_type,
    youtube_url,
    model_path,
    conf,
    infer_every,
    folder_files,
    webcam_index,
    video_file,
):
    yield from zone_monitor.stream(
        session_id,
        source_type,
        youtube_url,
        model_path,
        conf,
        infer_every,
        folder_files,
        webcam_index,
        video_file,
        browser_session_id,
    )


def load_classes(label_prompts):
    """라벨을 스캔해 (class_state, 최종 클래스 이름, 로드 기준 Tab2 프롬프트) 반환.

    이름은 Tab 2 프롬프트(label_prompts)를 우선으로 새로 구성한다.
    세 번째 반환값은 "이 프롬프트 기준으로 로드했다"는 추적용 값.
    """
    classes = dataset.scan_classes(label_prompts or "")
    joined = ", ".join(c["name"] for c in classes)
    return classes, joined, (label_prompts or "")


def refresh_model_dropdown():
    """학습된 모델 목록을 다시 스캔해 드롭다운 choices/value 를 갱신한다."""
    choices = models.list_trained_models()
    return gr.update(choices=choices, value=(choices[0][1] if choices else None))


def _refresh_webcam_dropdown(browser_devices_payload, channel: str):
    """서버와 현재 접속 브라우저의 카메라를 한 목록으로 합친다."""

    choices, value = webcams.refresh_webcam_dropdown(
        browser_devices_payload,
        channel,
    )
    server_count = sum(
        not str(choice_value).startswith("browser:")
        for _label, choice_value in choices
    )
    return (
        gr.update(choices=choices, value=value),
        webcams.browser_discovery_status(browser_devices_payload, server_count),
    )


def refresh_capture_webcams(browser_devices_payload):
    return _refresh_webcam_dropdown(browser_devices_payload, "capture")


def refresh_inference_webcams(browser_devices_payload):
    return _refresh_webcam_dropdown(browser_devices_payload, "inference")


def refresh_zone_webcams(browser_devices_payload):
    return _refresh_webcam_dropdown(browser_devices_payload, "zone")


def configure_browser_camera(browser_session_id, webcam_value, source_type):
    """접속 기기 카메라 선택 시 정확한 deviceId로 브라우저 입력을 연다."""

    if source_type != "웹캠":
        return gr.update(visible=False, value=None)
    try:
        source = webcams.parse_browser_webcam_value(
            webcam_value,
            browser_session_id,
        )
    except webcams.WebcamOpenError:
        source = None
    if source is None:
        return gr.update(visible=False, value=None)
    return gr.update(
        visible=True,
        value=None,
        webcam_options=gr.WebcamOptions(
            mirror=False,
            constraints={
                "video": {
                    "deviceId": {"exact": source.device_id},
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                },
                "audio": False,
            },
        ),
    )


def receive_browser_camera_frame(browser_session_id, webcam_value, frame_rgb):
    """모바일/원격 브라우저의 최신 카메라 프레임을 서버 버퍼로 전달한다."""

    webcams.push_browser_frame(browser_session_id, webcam_value, frame_rgb)



def _base_model_choices():
    """학습 탭 베이스 모델 드롭다운 choices — 첫 항목은 '처음부터'(값=빈 문자열)."""
    return [("yolo26n.pt (사전학습 · 처음부터)", "")] + models.list_trained_models()


def refresh_base_model_dropdown():
    """베이스 모델 목록 갱신 — 현재 선택은 유지(choices만 교체)."""
    return gr.update(choices=_base_model_choices())


def _dataset_choices(records):
    return [
        (
            f"{record['name']} · train {record['train_images']:,} / "
            f"val {record['val_images']:,} · {len(record['classes'])} classes",
            record["yaml"],
        )
        for record in (records or [])
    ]


def _dataset_table(records):
    return [
        [
            record["name"],
            record.get("source", "외부 데이터셋"),
            record["train_images"],
            record["val_images"],
            record["boxes"],
            record["missing_labels"],
            record["empty_labels"],
            ", ".join(record["classes"]),
            record["yaml"],
        ]
        for record in (records or [])
    ]


def _valid_dataset_selection(records, selected):
    available = {
        str(record["yaml"]).casefold(): record["yaml"] for record in (records or [])
    }
    result = []
    seen = set()
    for path in selected or []:
        key = str(path).casefold()
        if key in available and key not in seen:
            result.append(available[key])
            seen.add(key)
    return result


def _dataset_selection_text(records, selected):
    valid, message = dataset_importer.selection_summary(records, selected)
    return f"{'✅' if valid else '⚠️'} **학습 데이터셋:** {message}"


def _dataset_ui_result(records, selected, message):
    selected = _valid_dataset_selection(records, selected)
    summary = _dataset_selection_text(records, selected)
    return (
        records,
        gr.update(choices=_dataset_choices(records), value=selected),
        _dataset_table(records),
        message,
        summary,
        summary,
    )


def _downloaded_roboflow_choices():
    return [
        (item["label"], item["yaml"])
        for item in dataset_importer.list_downloaded_roboflow()
    ]


def add_archive_datasets(archive_files, records, selected):
    try:
        records, added = dataset_importer.register_archives(archive_files, records)
        selected = _valid_dataset_selection(records, selected)
        selected_keys = {str(path).casefold() for path in selected}
        for record in added:
            key = str(record["yaml"]).casefold()
            if key not in selected_keys:
                selected.append(record["yaml"])
                selected_keys.add(key)
        message = (
            f"✅ {len(added)}개 데이터셋을 압축 해제·검사하고 등록했습니다: "
            + ", ".join(record["name"] for record in added)
        )
    except dataset_importer.DatasetImportError as exc:
        message = f"❌ 등록 실패\n\n{exc}"
    return _dataset_ui_result(records or [], selected or [], message)


def add_roboflow_dataset(universe_url, records, selected):
    yield _dataset_ui_result(
        records or [],
        selected or [],
        "⏳ Roboflow 데이터셋을 다운로드하고 검사하는 중입니다...",
    ) + (gr.update(),)
    try:
        records, added, version = dataset_importer.download_roboflow_universe(
            universe_url,
            records,
        )
        selected = _valid_dataset_selection(records, selected)
        selected_keys = {str(path).casefold() for path in selected}
        for record in added:
            key = str(record["yaml"]).casefold()
            if key not in selected_keys:
                selected.append(record["yaml"])
                selected_keys.add(key)
        message = (
            f"✅ Roboflow v{version} 데이터셋을 다운로드·검사하고 등록했습니다: "
            + ", ".join(record["name"] for record in added)
        )
    except dataset_importer.DatasetImportError as exc:
        message = f"❌ Roboflow 다운로드 실패\n\n{exc}"
    yield _dataset_ui_result(records or [], selected or [], message) + (
        gr.update(choices=_downloaded_roboflow_choices(), value=None),
    )


def register_downloaded_roboflow(yaml_file, records, selected):
    """다운로드 목록 클릭 즉시 해당 데이터셋을 검사·등록한다."""
    loading = gr.update(visible=True)
    hidden = gr.update(visible=False)
    yield _dataset_ui_result(
        records or [],
        selected or [],
        "⏳ 선택한 Roboflow 데이터셋을 검사하고 등록하는 중입니다...",
    ) + (gr.update(), loading)
    try:
        records, added = dataset_importer.register_downloaded_roboflow(
            yaml_file,
            records,
        )
        selected = _valid_dataset_selection(records, selected)
        if str(added["yaml"]).casefold() not in {
            str(path).casefold() for path in selected
        }:
            selected.append(added["yaml"])
        message = f"✅ Roboflow 데이터셋을 검사하고 등록했습니다: {added['name']}"
    except dataset_importer.DatasetImportError as exc:
        message = f"❌ Roboflow 데이터셋 등록 실패\n\n{exc}"
    yield _dataset_ui_result(records or [], selected or [], message) + (
        gr.update(value=None), hidden,
    )


def refresh_downloaded_roboflow():
    choices = _downloaded_roboflow_choices()
    message = (
        f"✅ 다운로드된 Roboflow 데이터셋 {len(choices)}개를 찾았습니다."
        if choices
        else "ℹ️ 다운로드된 Roboflow 데이터셋이 없습니다."
    )
    return gr.update(choices=choices, value=None), message


def refresh_datasets(records, selected):
    records, errors = dataset_importer.refresh_registry(records)
    message = f"✅ 등록된 데이터셋 {len(records)}개를 다시 검사했습니다."
    if errors:
        message += "\n\n⚠️ 사용할 수 없어 제외한 항목:\n" + "\n".join(
            f"- {error}" for error in errors
        )
    return _dataset_ui_result(records, selected, message)


def remove_external_datasets(records, selected):
    selected_set = {str(path).casefold() for path in (selected or [])}
    default_yaml = str(dataset_importer.DEFAULT_DATASET_YAML.resolve()).casefold()
    removed = [
        record
        for record in (records or [])
        if str(record["yaml"]).casefold() in selected_set
        and str(record["yaml"]).casefold() != default_yaml
    ]
    records = [
        record
        for record in (records or [])
        if record not in removed
    ]
    remaining_selection = [
        path
        for path in (selected or [])
        if str(path).casefold() not in {
            str(record["yaml"]).casefold() for record in removed
        }
    ]
    message = (
        f"✅ 외부 데이터셋 {len(removed)}개를 현재 목록에서 제거했습니다."
        if removed
        else "ℹ️ 제거할 외부 데이터셋이 선택되지 않았습니다."
    )
    return _dataset_ui_result(records, remaining_selection, message)


def update_dataset_selection_status(records, selected):
    summary = _dataset_selection_text(records, selected)
    return summary, summary


def run_label(prompts_str, conf, selected_sources):
    yield from labeler.label(prompts_str, float(conf), selected_sources)


def on_gallery_select(prompts_str, filter_empty, evt: gr.SelectData):
    """갤러리 선택 이벤트 래퍼 — SelectData 어노테이션으로 evt 주입을 명시."""
    return dataset.select_frame(prompts_str, filter_empty, evt)


def on_zone_editor_select(session_id, mode, evt: gr.SelectData):
    """고정 편집 프레임의 자연 크기 좌표를 영역 서비스에 전달한다."""
    return zone_monitor.select_editor_point(session_id, mode, evt.index)


def run_label_preview(prompts_str, conf, n_preview, selected_sources):
    yield from labeler.preview(
        prompts_str,
        float(conf),
        int(n_preview),
        selected_sources,
    )


def refresh_label_source_choices(current_selection=None):
    choices = labeler.source_choices()
    values = [value for _label, value in choices]
    selected = [value for value in (current_selection or []) if value in values]
    if not selected:
        selected = values
    return gr.update(choices=choices, value=selected)


def refresh_validation_source_choices(current_selection=None):
    choices = dataset.source_choices()
    values = {value for _label, value in choices}
    selected = [value for value in (current_selection or []) if value in values]
    return gr.update(choices=choices, value=selected)


_WRAP_STYLE = (
    "height:420px;overflow-y:scroll;display:flex;flex-direction:column-reverse;"
    "background:#1e1e1e;border-radius:6px;"
)
_PRE_STYLE = (
    "margin:0;padding:10px 14px;color:#d4d4d4;"
    "font-family:'Consolas','Courier New',monospace;font-size:12px;"
    "white-space:pre-wrap;word-break:break-all;"
)


def run_train(
    epochs,
    imgsz,
    batch,
    patience,
    device,
    name,
    base_model,
    dataset_yamls=None,
):
    for _text in trainer.train(
        epochs,
        imgsz,
        batch,
        patience,
        device,
        name,
        base_model,
        dataset_yamls,
    ):
        yield training_panel_update()


_QUALITY_METRICS = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("map50", "mAP50"),
    ("map50_95", "mAP50-95"),
)
_LOSS_METRICS = (
    ("box_loss", "Train Box"),
    ("val_box_loss", "Val Box"),
    ("cls_loss", "Train Class"),
    ("val_cls_loss", "Val Class"),
    ("dfl_loss", "Train DFL"),
    ("val_dfl_loss", "Val DFL"),
)


def _chart_data(history, series, *, percent: bool = False) -> pd.DataFrame:
    rows = []
    for item in history or []:
        epoch = item.get("epoch")
        for key, label in series:
            value = item.get(key)
            if epoch is None or value is None:
                continue
            rows.append(
                {
                    "Epoch": int(epoch),
                    "Metric": label,
                    "Value": float(value) * (100.0 if percent else 1.0),
                }
            )
    return pd.DataFrame(rows, columns=["Epoch", "Metric", "Value"])


def _format_duration(seconds) -> str:
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "-"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


def _training_progress(snapshot, history) -> str:
    total = max(0, int(snapshot.get("total_epochs") or 0))
    current = int(
        (history[-1].get("epoch") if history else None)
        or snapshot.get("current_epoch")
        or 0
    )
    percent = min(100.0, current / total * 100.0) if total else 0.0

    best_text = "best mAP50-95 -"
    if history:
        valid = [
            item for item in history if item.get("map50_95") is not None
        ]
        if valid:
            best = max(valid, key=lambda item: item["map50_95"])
            best_text = (
                f"best mAP50-95 {best['map50_95'] * 100:.1f}% "
                f"(epoch {best['epoch']})"
            )

    elapsed = history[-1].get("time") if history else None
    eta = None
    if elapsed and current and total > current:
        eta = float(elapsed) / current * (total - current)
    state = _html.escape(str(snapshot.get("message") or "학습 대기 중"))
    return (
        '<div class="train-progress-visual">'
        f'<div class="train-progress-label"><strong>{state}</strong>'
        f'<span>Epoch {current}/{total or "-"} · {percent:.0f}%</span></div>'
        f'<progress value="{percent:.2f}" max="100" aria-label="학습 진행률">'
        f'{percent:.0f}%</progress>'
        '<div class="train-progress-detail">'
        f'<span>{best_text}</span><span>경과 {_format_duration(elapsed)}</span>'
        f'<span>예상 잔여 {_format_duration(eta)}</span></div></div>'
    )


def _training_artifacts(run_dir) -> list[tuple[str, str]]:
    if not run_dir:
        return []
    root = Path(run_dir)
    candidates = [
        ("results.png", "전체 학습 결과"),
        ("confusion_matrix_normalized.png", "정규화 혼동행렬"),
        ("BoxPR_curve.png", "클래스별 Precision-Recall"),
        ("BoxF1_curve.png", "클래스별 F1-Confidence"),
        ("val_batch0_pred.jpg", "Validation 예측 샘플"),
    ]
    return [
        (str(root / filename), caption)
        for filename, caption in candidates
        if (root / filename).is_file()
    ]


def training_panel_update():
    snapshot = trainer.training_snapshot()
    active = bool(snapshot.get("active"))
    status = snapshot.get("status", "idle")
    icon = {
        "preparing": "⏳",
        "running": "🟠",
        "stopping": "⏹️",
        "completed": "✅",
        "stopped": "⏹️",
        "error": "❌",
        "interrupted": "⚠️",
    }.get(status, "⚪")
    details = [f"{icon} **{snapshot.get('message', '학습 대기 중')}**"]
    if snapshot.get("run_name"):
        details.append(f"모델: `{snapshot['run_name']}`")
    if snapshot.get("run_dir"):
        details.append(f"결과: `{snapshot['run_dir']}`")
    if snapshot.get("started_at"):
        details.append(f"시작: `{snapshot['started_at']}`")
    status_markdown = "  \n".join(details)

    log_text = snapshot.get("log") or "학습 로그가 아직 없습니다."
    escaped = _html.escape(log_text)
    log_html = f'<div style="{_WRAP_STYLE}"><pre style="{_PRE_STYLE}">{escaped}</pre></div>'
    history = snapshot.get("history") or []
    return (
        status_markdown,
        _training_progress(snapshot, history),
        _chart_data(history, _QUALITY_METRICS, percent=True),
        _chart_data(history, _LOSS_METRICS),
        _training_artifacts(snapshot.get("run_dir")),
        log_html,
        gr.update(
            value="학습 진행 중" if active else "학습 시작",
            interactive=not active,
        ),
        gr.update(interactive=active),
    )


def stop_train():
    trainer.stop()
    return training_panel_update()


_BROWSER_CAMERA_DISCOVERY_JS = r"""async (_previous) => {
    const result = {devices: [], error: ""};
    if (!window.isSecureContext) {
        result.error = "접속 기기 카메라는 HTTPS(또는 localhost)에서만 사용할 수 있습니다.";
        return JSON.stringify(result);
    }
    if (!navigator.mediaDevices?.getUserMedia || !navigator.mediaDevices?.enumerateDevices) {
        result.error = "이 브라우저는 카메라 검색을 지원하지 않습니다.";
        return JSON.stringify(result);
    }
    let probe = null;
    try {
        probe = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
        const devices = await navigator.mediaDevices.enumerateDevices();
        result.devices = devices
            .filter((device) => device.kind === "videoinput" && device.deviceId)
            .map((device, index) => ({
                id: device.deviceId,
                label: device.label || `카메라 ${index + 1}`,
            }));
    } catch (error) {
        const name = error?.name || "";
        if (name === "NotAllowedError" || name === "SecurityError") {
            result.error = "카메라 권한이 거부되었습니다. 브라우저 사이트 권한에서 카메라를 허용하세요.";
        } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
            result.error = "접속 기기에서 사용할 수 있는 카메라를 찾지 못했습니다.";
        } else if (name === "NotReadableError" || name === "TrackStartError") {
            result.error = "접속 기기 카메라가 다른 앱에서 사용 중이거나 열 수 없습니다.";
        } else {
            result.error = `접속 기기 카메라 검색 실패: ${error?.message || name || "알 수 없는 오류"}`;
        }
    } finally {
        probe?.getTracks().forEach((track) => track.stop());
    }
    return JSON.stringify(result);
}"""


_AUTO_START_INFERENCE_CAMERA_JS = r"""(
    browserSessionId,
    modelPath,
    sourceType,
    youtubeUrl,
    conf,
    inferEvery,
    folderFiles,
    webcamValue,
    videoFile
) => {
    if (sourceType === "웹캠" && String(webcamValue || "").startsWith("browser:")) {
        const startIcon = document.querySelector(
            '#inference_browser_camera [title="start recording"]'
        );
        const startButton = startIcon?.closest("button");
        startButton?.click();
    }
    return [
        browserSessionId,
        modelPath,
        sourceType,
        youtubeUrl,
        conf,
        inferEvery,
        folderFiles,
        webcamValue,
        videoFile,
    ];
}"""


_CSS = (
    ".src-hidden { display: none !important; }"
    ".cls-name-box input { font-weight:600; }"
    # 본문 패널이 좁은 화면에서도 넘치지 않도록
    ".main-panel { max-width: 1100px; margin: 0 auto; width: 100%; }"
    # Gradio 이미지 전체화면은 내부 프레임 폭을 원본 이미지 크기에 맞추므로 저해상도
    # 프레임이 작게 남는다. 실시간 미리보기는 화면 전체를 사용하되 비율은 보존한다.
    ".live-preview.fullscreen .image-container, "
    ".live-preview .image-container:fullscreen { "
    "width: 100vw !important; height: 100vh !important; background: #000; }"
    ".live-preview.fullscreen .image-container > button, "
    ".live-preview.fullscreen .image-container .image-frame, "
    ".live-preview .image-container:fullscreen > button, "
    ".live-preview .image-container:fullscreen .image-frame { "
    "width: 100% !important; height: 100% !important; }"
    ".live-preview.fullscreen .image-container img, "
    ".live-preview .image-container:fullscreen img { "
    "width: 100% !important; height: 100% !important; "
    "max-width: 100vw !important; max-height: 100vh !important; "
    "object-fit: contain !important; }"
    # 브라우저 웹캠은 추론 시작 시 자동 스트리밍하므로 Gradio의 수동 '녹음' UI를 숨긴다.
    "#inference_browser_camera .button-wrap { display: none !important; }"
    # 원본 브라우저 영상을 계속 재생하고 추론 결과 SVG만 그 위에 겹친다.
    "#inference_browser_stage { position: relative; isolation: isolate; }"
    "#inference_browser_overlay { position: absolute !important; inset: 0; z-index: 5; "
    "pointer-events: none; padding: 0 !important; border: 0 !important; "
    "background: transparent !important; }"
    "#inference_browser_overlay .html-container, #inference_browser_overlay .prose, "
    "#inference_browser_overlay svg { "
    "display: block; width: 100%; height: 100%; margin: 0; padding: 0; }"
    # 미리보기 갤러리: 항상 4열 격자로 고정. Gradio는 이미지 수에 맞춰 --grid-cols를
    # 줄여(1장이면 1열) 이미지가 칸 전체로 커지므로, grid-template-columns를 4열로 강제.
    "#label_gallery .grid-container { grid-template-columns: repeat(4, minmax(0, 1fr)) !important; }"
    ".sample-card button { min-height: 72px; border: 1px solid #d4d4d8 !important; "
    "background: #fff !important; color: #18181b !important; text-align: left; "
    "justify-content: flex-start; box-shadow: 0 1px 2px rgba(0,0,0,.06); }"
    "@keyframes rf-spin { to { transform: rotate(360deg); } }"
    ".rf-loading { display:flex;align-items:center;gap:10px;padding:10px 12px;"
    "border-radius:8px;background:rgba(249,115,22,.12); }"
    ".rf-spinner { width:20px;height:20px;border:3px solid rgba(249,115,22,.25);"
    "border-top-color:#f97316;border-radius:50%;animation:rf-spin .8s linear infinite; }"
    ".train-progress-visual { display:grid;gap:8px;padding:4px 0 8px; }"
    ".train-progress-label,.train-progress-detail { display:flex;gap:12px;"
    "justify-content:space-between;flex-wrap:wrap; }"
    ".train-progress-detail { opacity:.75;font-size:.9em; }"
    ".train-progress-visual progress { width:100%;height:18px;accent-color:#2563eb; }"
)

_webcam_choices, _webcam_value = webcams.refresh_webcam_dropdown()


with gr.Blocks(title="YOLO 파이프라인") as demo:

    browser_session_id = gr.State(
        value=webcams.create_browser_session,
        time_to_live=3600,
        delete_callback=webcams.delete_browser_session,
    )
    gr.Markdown("# YOLO 파이프라인")

    with gr.Tabs(elem_classes="main-panel"):
        # ── 프레임 추출 ──────────────────────────────────────
        with gr.Tab("프레임 추출"):
            gr.Markdown("### 소스 선택 & 프레임 추출 (중지 버튼으로 종료)")

            with gr.Row():
                source_type = gr.Radio(
                    choices=["웹캠", "YouTube URL", "비디오 파일", "이미지 폴더"],
                    value="웹캠",
                    label="소스",
                )
                capture_fps = gr.Slider(
                    minimum=1, maximum=30, value=5, step=1,
                    label="캡처 FPS (초당 저장 프레임 수)",
                    info="이미지 폴더 선택 시 무시됨",
                )

            youtube_url = gr.Textbox(
                placeholder=(
                    "https://www.youtube.com/watch?v=AAA\n"
                    "https://www.youtube.com/watch?v=BBB"
                ),
                label="YouTube URL 목록 (한 줄에 하나)",
                lines=5,
                elem_id="tab1_youtube_url",
                elem_classes=["src-hidden"],
            )
            with gr.Row(elem_id="tab1_youtube_sample_row", elem_classes=["src-hidden"]):
                youtube_sample_btn = gr.Button(
                    "샘플 YouTube URL 사용\nsamples/sample_url.txt",
                    elem_classes=["sample-card"],
                )
            with gr.Row(elem_id="tab1_webcam_row"):
                webcam_index = gr.Dropdown(
                    choices=_webcam_choices,
                    value=_webcam_value,
                    allow_custom_value=True,
                    label="웹캠 장비",
                    info="서버 카메라가 먼저, 현재 접속 기기의 카메라가 다음에 표시됩니다.",
                    scale=4,
                )
                webcam_refresh = gr.Button("서버 + 접속 기기 카메라 검색", scale=1)
            webcam_browser_payload = gr.Textbox(visible=False)
            webcam_browser_input = gr.Image(
                label="접속 기기 카메라 입력 — 스트리밍 시작 후 아래의 캡처 시작을 누르세요",
                sources=["webcam"],
                type="numpy",
                streaming=True,
                interactive=True,
                visible=False,
                height=360,
                elem_classes=["live-preview"],
                webcam_options=gr.WebcamOptions(mirror=False),
            )
            with gr.Row(elem_id="tab1_video_row", elem_classes=["src-hidden"]):
                video_file = gr.File(
                    label="비디오 파일 업로드",
                    file_count="single",
                    type="filepath",
                    file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"],
                    height=120,
                )
            with gr.Row(elem_id="tab1_video_sample_row", elem_classes=["src-hidden"]):
                video_sample_btn = gr.Button(
                    "샘플 비디오 파일 사용\nsamples/sample.mp4",
                    elem_classes=["sample-card"],
                )
            with gr.Row(elem_id="tab1_folder_row", elem_classes=["src-hidden"]) as folder_row:
                folder_files = gr.File(
                    label="이미지 폴더 업로드 (폴더 또는 여러 파일 선택)",
                    file_count="directory",
                    type="filepath",
                    height=200,
                )
            with gr.Row(elem_id="tab1_folder_sample_row", elem_classes=["src-hidden"]):
                folder_sample_btn = gr.Button(
                    "샘플 이미지 폴더 사용\nsamples/sample_image",
                    elem_classes=["sample-card"],
                )

            source_type.change(
                fn=None,
                inputs=source_type,
                outputs=None,
                js="""(s) => {
                    const yt = document.getElementById('tab1_youtube_url');
                    const ys = document.getElementById('tab1_youtube_sample_row');
                    const wc = document.getElementById('tab1_webcam_row');
                    const vf = document.getElementById('tab1_video_row');
                    const vs = document.getElementById('tab1_video_sample_row');
                    const fr = document.getElementById('tab1_folder_row');
                    const fs = document.getElementById('tab1_folder_sample_row');
                    if (yt) yt.classList.toggle('src-hidden', s !== 'YouTube URL');
                    if (ys) ys.classList.toggle('src-hidden', s !== 'YouTube URL');
                    if (wc) wc.classList.toggle('src-hidden', s !== '웹캠');
                    if (vf) vf.classList.toggle('src-hidden', s !== '비디오 파일');
                    if (vs) vs.classList.toggle('src-hidden', s !== '비디오 파일');
                    if (fr) fr.classList.toggle('src-hidden', s !== '이미지 폴더');
                    if (fs) fs.classList.toggle('src-hidden', s !== '이미지 폴더');
                }""",
            )
            webcam_index.change(
                fn=configure_browser_camera,
                inputs=[browser_session_id, webcam_index, source_type],
                outputs=webcam_browser_input,
                show_progress="hidden",
            )
            source_type.change(
                fn=configure_browser_camera,
                inputs=[browser_session_id, webcam_index, source_type],
                outputs=webcam_browser_input,
                show_progress="hidden",
            )
            webcam_browser_input.stream(
                fn=receive_browser_camera_frame,
                inputs=[browser_session_id, webcam_index, webcam_browser_input],
                outputs=None,
                queue=False,
                show_progress="hidden",
                stream_every=0.1,
            )
            with gr.Row():
                cap_start_btn = gr.Button("캡처 시작", variant="primary")
                cap_stop_btn  = gr.Button("중지", variant="stop")

            cap_preview = gr.Image(
                label="실시간 미리보기",
                type="numpy",
                streaming=True,
                elem_classes=["live-preview"],
            )
            cap_status  = gr.Textbox(label="상태", interactive=False)

            youtube_sample_btn.click(
                fn=load_sample_youtube_url,
                outputs=[youtube_url, cap_status],
            )
            video_sample_btn.click(
                fn=load_sample_video_file,
                outputs=[video_file, cap_status],
            )
            folder_sample_btn.click(
                fn=load_sample_image_folder,
                outputs=[folder_files, cap_status],
            )

            cap_prepare_event = cap_start_btn.click(
                fn=prepare_capture_preview,
                outputs=[cap_preview, cap_status],
            )
            capture_event = cap_prepare_event.then(
                fn=run_capture,
                inputs=[browser_session_id, source_type, youtube_url, capture_fps, folder_files, webcam_index, video_file],
                outputs=[cap_preview, cap_status],
                show_progress="hidden",
                concurrency_limit=1,
                concurrency_id="frame_capture",
            )
            cap_stop_btn.click(
                fn=stop_capture_and_reset,
                outputs=[cap_preview, cap_status],
                cancels=[capture_event],
            )
            webcam_refresh.click(
                fn=refresh_capture_webcams,
                inputs=webcam_browser_payload,
                outputs=[webcam_index, cap_status],
                js=_BROWSER_CAMERA_DISCOVERY_JS,
            )

        # ── SAM3 오토라벨링 ──────────────────────────────────
        with gr.Tab("SAM3 오토라벨링") as panel_labeler:
            gr.Markdown("### SAM3 텍스트 프롬프트로 자동 라벨링")
            gr.Markdown(
                "프롬프트와 conf를 입력하고 **① 미리보기**로 샘플 라벨을 확인하세요. "
                "결과가 괜찮으면 **② 전체 라벨링 시작**으로 모든 프레임에 적용합니다."
            )

            prompts_input = gr.Textbox(
                placeholder="white hardhat, blue safety vest",
                label="클래스 프롬프트 (쉼표 구분)",
                value="white hardhat, blue safety vest",
            )
            initial_label_sources = labeler.source_choices()
            label_source_select = gr.CheckboxGroup(
                choices=initial_label_sources,
                value=[value for _label, value in initial_label_sources],
                label="라벨링할 소스",
                info="선택한 URL·웹캠 세션만 처리하며 다른 소스의 기존 라벨은 보존합니다.",
            )
            with gr.Row():
                conf_slider = gr.Slider(
                    minimum=0.05, maximum=0.9, value=0.25, step=0.05,
                    label="신뢰도 임계값 (conf)",
                )
                n_preview_slider = gr.Slider(
                    minimum=1, maximum=12, value=4, step=1,
                    label="미리보기 샘플 수",
                    info="전체 프레임에서 균등 샘플링해 라벨 결과를 미리 확인 (저장 안 됨)",
                )

            with gr.Row():
                label_preview_btn = gr.Button("① 미리보기", variant="secondary")
                label_start_btn   = gr.Button("② 전체 라벨링 시작", variant="primary")
                label_stop_btn    = gr.Button("중지", variant="stop")

            label_gallery = gr.Gallery(
                label="미리보기 결과 (마스크 오버레이 · 저장 전)",
                columns=4,
                height=360,
                object_fit="contain",
                elem_id="label_gallery",
            )

            label_preview = gr.Image(
                label="전체 라벨링 진행 미리보기", type="numpy", streaming=True,
                elem_classes=["live-preview"],
            )
            label_status  = gr.Textbox(label="상태", interactive=False)

            preview_event = label_preview_btn.click(
                fn=run_label_preview,
                inputs=[
                    prompts_input,
                    conf_slider,
                    n_preview_slider,
                    label_source_select,
                ],
                outputs=[label_gallery, label_status],
            )
            label_event = label_start_btn.click(
                fn=run_label,
                inputs=[prompts_input, conf_slider, label_source_select],
                outputs=[label_preview, label_status],
            )
            label_stop_btn.click(
                fn=labeler.stop,
                cancels=[preview_event, label_event],
            )


        # ── 데이터셋 검토 & 구성 ────────────────────────────────
        with gr.Tab("데이터셋 구성") as panel_dataset:
            gr.Markdown("### 라벨링 결과 검토 후 train/val 분할")

            gr.Markdown(
                "#### 클래스 목록 — 각 이름 칸에서 바로 수정하세요\n"
                "라벨에서 감지된 클래스 ID와 그 클래스가 들어있는 프레임 수입니다. "
                "**이름은 이 칸에서 직접 수정**하면 데이터셋 구성·미리보기에 그대로 반영됩니다. "
                "Roboflow 공사장 데이터와 통합할 때는 각각 `Hardhat`, `Safety Vest`로 "
                "정확히 맞추세요."
            )

            ds_class_state = gr.State([])
            # 마지막 로드에 사용한 Tab 2 프롬프트 — 변경 감지(재라벨링 시 자동 갱신)용
            ds_last_label  = gr.State("")

            load_classes_btn = gr.Button("클래스 불러오기 / 새로고침", variant="secondary")

            @gr.render(inputs=ds_class_state)
            def _render_class_editor(classes):
                if not classes:
                    gr.Markdown(
                        "_「클래스 불러오기」를 누르면 라벨에서 감지된 클래스 목록이 표시됩니다. "
                        "(Tab 2 오토라벨링을 먼저 실행하세요.)_"
                    )
                    return

                name_boxes = []
                for c in classes:
                    with gr.Row(equal_height=True):
                        gr.Markdown(f"**ID {c['id']}** · {c['count']}개 프레임")
                        nb = gr.Textbox(
                            value=c["name"],
                            show_label=False,
                            container=False,
                            scale=3,
                            interactive=True,
                            elem_classes=["cls-name-box"],
                        )
                    name_boxes.append(nb)

                def _sync(*names):
                    # ID 순서대로 이름을 모아 최종 클래스 이름 문자열 생성
                    return ", ".join((n or "").strip() for n in names)

                # 어느 칸을 수정해도 즉시 집계 State(ds_prompts)에 반영
                # (ds_class_state 미변경 → 리렌더/포커스 유실 없음)
                for nb in name_boxes:
                    nb.change(_sync, inputs=name_boxes, outputs=ds_prompts)

            # 클래스 이름 칸들에서 집계한 최종 클래스 이름(ID 0,1,2… 순서, 쉼표 결합).
            # 화면에는 표시하지 않고 미리보기/구성 등 다운스트림의 입력 소스로만 사용.
            ds_prompts = gr.State("")

            with gr.Row():
                filter_empty_chk = gr.Checkbox(
                    label="라벨 없는 프레임 숨기기 / 제외",
                    value=False,
                )
                val_ratio_slider = gr.Slider(
                    minimum=0.1, maximum=0.4, value=0.2, step=0.05,
                    label="자동 분리 시 Validation 목표 비율",
                )

            initial_validation_sources = dataset.source_choices()
            validation_source_select = gr.CheckboxGroup(
                choices=initial_validation_sources,
                value=[],
                label="Validation으로 사용할 소스",
                info=(
                    "선택한 소스 전체를 validation으로 배정합니다. "
                    "비워두면 목표 비율에 맞춰 소스 단위로 자동 선택합니다."
                ),
            )
            validation_source_refresh = gr.Button(
                "소스 목록 새로고침",
                variant="secondary",
            )

            preview_btn = gr.Button("라벨 미리보기 로드", variant="secondary")
            ds_stats    = gr.Textbox(label="통계", interactive=False)
            ds_gallery  = gr.Gallery(
                label="프레임별 라벨 확인 — 이미지를 클릭하면 우측 상세 보기 + 삭제 가능",
                columns=4,
                height=500,
                object_fit="contain",
            )

            preview_btn.click(
                fn=dataset.load_preview,
                inputs=[ds_prompts, filter_empty_chk],
                outputs=[ds_gallery, ds_stats],
            )

            # ── 상세 보기 & 삭제 ────────────────────────────────────────
            selected_stem_state = gr.State(value="")
            with gr.Row():
                ds_detail = gr.Image(
                    label="선택한 이미지 상세 보기",
                    type="numpy",
                    interactive=False,
                    visible=True,
                    height=400,
                    scale=2,
                )
                with gr.Column(scale=1):
                    delete_status = gr.Textbox(
                        label="선택 / 삭제 상태", interactive=False, lines=3,
                    )
                    delete_btn = gr.Button("🗑 선택한 이미지 삭제", variant="stop")

            ds_gallery.select(
                fn=on_gallery_select,
                inputs=[ds_prompts, filter_empty_chk],
                outputs=[ds_detail, selected_stem_state, delete_status],
            )
            delete_btn.click(
                fn=dataset.delete_frame,
                inputs=[selected_stem_state, ds_prompts, filter_empty_chk],
                outputs=[ds_gallery, ds_stats, ds_detail, selected_stem_state, delete_status],
            )

            gr.Markdown("---")
            build_btn    = gr.Button("데이터셋 구성 (train/val 분할)", variant="primary")
            build_status = gr.Textbox(label="진행 상태", interactive=False)

            build_btn.click(
                fn=dataset.build_dataset,
                inputs=[
                    ds_prompts,
                    val_ratio_slider,
                    filter_empty_chk,
                    validation_source_select,
                ],
                outputs=build_status,
            )

            validation_source_refresh.click(
                fn=refresh_validation_source_choices,
                inputs=validation_source_select,
                outputs=validation_source_select,
            )

            load_classes_btn.click(
                fn=load_classes,
                inputs=[prompts_input],
                outputs=[ds_class_state, ds_prompts, ds_last_label],
            )

        # ── 외부 데이터셋 불러오기 ───────────────────────────
        with gr.Tab("데이터셋 불러오기") as panel_dataset_import:
            gr.Markdown("### 학습 데이터셋 불러오기")
            gr.Markdown(
                "기존 **데이터셋 검토 & 구성** 단계에서 만든 데이터셋과 외부 YOLO 탐지 "
                "데이터셋을 함께 관리합니다. 클래스 구성이 다른 데이터셋도 원본 라벨을 "
                "건드리지 않고 학습용 복사본에서 클래스 ID를 자동 통합합니다."
            )

            initial_dataset_records = dataset_importer.initial_registry()
            initial_dataset_selection = [
                record["yaml"] for record in initial_dataset_records
            ]
            dataset_registry_state = gr.State(value=initial_dataset_records)

            with gr.Group():
                gr.Markdown("#### Roboflow Universe에서 바로 다운로드")
                roboflow_universe_url = gr.Textbox(
                    label="Roboflow Universe 데이터셋 URL",
                    placeholder=(
                        "https://universe.roboflow.com/workspace/project "
                        "또는 .../dataset/4"
                    ),
                )
                roboflow_dataset_add = gr.Button(
                    "YOLO26 데이터셋 다운로드 후 등록",
                    variant="primary",
                )
                gr.Markdown(
                    "> 프로젝트 URL은 최신 버전을 자동 선택합니다. `.env`의 "
                    "`ROBOFLOW_API_KEY`를 사용하며 다운로드 결과는 "
                    "`dataset/external/roboflow/`에 저장됩니다."
                )

            with gr.Group():
                gr.Markdown("#### 다운로드된 Roboflow 데이터셋")
                gr.Markdown(
                    "항목을 클릭하면 데이터셋 전체를 검사한 뒤 등록하고, "
                    "학습 데이터셋 선택에도 자동으로 추가합니다."
                )
                with gr.Row():
                    roboflow_downloaded_select = gr.Radio(
                        choices=_downloaded_roboflow_choices(),
                        value=None,
                        label="다운로드 목록 (클릭하여 등록)",
                        scale=4,
                    )
                    roboflow_downloaded_refresh = gr.Button(
                        "다운로드 목록 새로고침",
                        scale=1,
                    )
                roboflow_register_loading = gr.HTML(
                    '<div class="rf-loading"><span class="rf-spinner"></span>'
                    "<span>데이터셋을 검사하고 등록하는 중입니다…</span></div>",
                    visible=False,
                )

            gr.Markdown("#### 로컬 ZIP에서 등록")

            dataset_archives = gr.File(
                label="YOLO 데이터셋 ZIP 업로드 (여러 개 가능)",
                file_count="multiple",
                type="filepath",
                file_types=[".zip"],
                height=180,
            )
            archive_dataset_add = gr.Button(
                "ZIP 압축 해제 후 등록",
                variant="primary",
            )

            gr.Markdown(
                "> ZIP은 각 데이터셋의 `data.yaml`, `train/images`, `train/labels`, "
                "`valid/images`, `valid/labels` 같은 폴더 구조를 포함해야 합니다."
            )

            with gr.Row():
                dataset_refresh_btn = gr.Button("등록 목록 다시 검사")
                dataset_remove_btn = gr.Button(
                    "선택한 외부 데이터셋을 목록에서 제거",
                    variant="stop",
                )

            training_dataset_select = gr.CheckboxGroup(
                choices=_dataset_choices(initial_dataset_records),
                value=initial_dataset_selection,
                label="학습에 사용할 데이터셋",
                info="체크한 데이터셋을 한 번의 학습에 함께 적용합니다. "
                     "클래스 구성이 다르면 통합 클래스 목록으로 라벨 ID를 자동 재매핑합니다.",
            )
            initial_dataset_summary = _dataset_selection_text(
                initial_dataset_records,
                initial_dataset_selection,
            )
            dataset_selection_status = gr.Markdown(initial_dataset_summary)
            dataset_import_status = gr.Markdown(
                "YOLO 데이터셋 ZIP을 하나 이상 업로드하세요."
            )
            dataset_registry_table = gr.Dataframe(
                value=_dataset_table(initial_dataset_records),
                headers=[
                    "데이터셋",
                    "소스",
                    "Train",
                    "Val",
                    "BBox",
                    "라벨 누락",
                    "빈 라벨",
                    "클래스",
                    "YAML",
                ],
                datatype=[
                    "str",
                    "str",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "str",
                    "str",
                ],
                interactive=False,
                label="등록된 데이터셋 검사 결과",
                wrap=True,
            )

        # ── YOLO 학습 ────────────────────────────────────────
        with gr.Tab("YOLO 학습") as panel_train:
            gr.Markdown("### YOLO 모델 학습")
            train_dataset_summary = gr.Markdown(initial_dataset_summary)
            gr.Markdown(
                "학습 데이터 변경은 **데이터셋 불러오기** 탭에서 선택하세요."
            )
            gr.Markdown(
                "> **파라미터 안내** — 아래 값은 ultralytics 기본값 기준이며 최적화된 값이 아닙니다. "
                "데이터셋 크기·GPU 환경에 따라 조정하세요."
            )

            _base_models = _base_model_choices()
            with gr.Row():
                base_model_dd = gr.Dropdown(
                    choices=_base_models,
                    value="",
                    label="베이스 모델 (이어학습)",
                    info="비우면 사전학습 yolo26n.pt로 처음부터 학습. 학습된 모델을 고르면 "
                         "그 가중치 위에 이어서 파인튜닝합니다 (괄호 안은 생성 날짜).",
                    scale=4,
                )
                base_model_refresh = gr.Button("모델 목록 새로고침", scale=1)

            train_name = gr.Textbox(
                value="train",
                label="모델(결과) 이름",
                placeholder="train",
                info="결과 저장 폴더 이름 → runs/detect/<이름>/weights/best.pt. "
                     "추론·침입 감지 탭의 모델 목록에 이 이름으로 표시됩니다. "
                     "같은 이름이 이미 있으면 자동으로 숫자가 붙습니다(덮어쓰지 않음).",
            )

            with gr.Row():
                epochs_slider = gr.Slider(
                    minimum=1, maximum=300, value=50, step=1,
                    label="Epochs",
                    info="전체 데이터를 몇 번 반복 학습할지. 프레임 500장 기준 50~100이 적당. "
                         "너무 높으면 overfitting 위험 → 학습 후 loss 곡선 확인 권장",
                )
                imgsz_slider = gr.Slider(
                    minimum=320, maximum=1280, value=640, step=32,
                    label="Image Size",
                    info="입력 이미지 리사이즈 크기. 640이 표준. "
                         "객체가 작거나 원본 해상도가 낮으면 416~480도 고려 가능",
                )

            with gr.Row():
                batch_slider = gr.Slider(
                    minimum=1, maximum=64, value=16, step=1,
                    label="Batch Size",
                    info="한 번에 처리할 이미지 수. GPU VRAM 8GB 이하면 8로 낮추세요. "
                         "CUDA OOM 오류 발생 시 가장 먼저 줄일 값",
                )
                patience_slider = gr.Slider(
                    minimum=1, maximum=100, value=10, step=1,
                    label="Early Stopping Patience",
                    info="검증 성능이 연속으로 개선되지 않아도 기다릴 epoch 수. "
                         "10이면 10 epoch 연속 개선이 없을 때 자동 종료합니다.",
                )

            device_radio = gr.Radio(
                choices=["auto", "mps", "cpu", "0"],
                value="mps",
                label="Device",
                info="mps: Apple Silicon GPU / auto: CUDA GPU가 없으면 CPU / "
                     "cpu: CPU 강제 (느림) / 0: 첫 번째 CUDA GPU",
            )

            with gr.Row():
                train_start_btn = gr.Button("학습 시작", variant="primary")
                train_stop_btn  = gr.Button("중지", variant="stop", interactive=False)

            train_runtime_status = gr.Markdown("⚪ **학습 대기 중**")
            train_epoch_progress = gr.HTML(
                _training_progress(
                    {"message": "학습 대기 중", "total_epochs": 0},
                    [],
                )
            )
            with gr.Row():
                train_quality_chart = gr.LinePlot(
                    value=_chart_data([], _QUALITY_METRICS, percent=True),
                    x="Epoch",
                    y="Value",
                    color="Metric",
                    title="검증 성능 변화",
                    x_title="Epoch",
                    y_title="성능 (%)",
                    y_lim=[0, 100],
                    tooltip="axis",
                    height=360,
                )
                train_loss_chart = gr.LinePlot(
                    value=_chart_data([], _LOSS_METRICS),
                    x="Epoch",
                    y="Value",
                    color="Metric",
                    title="Train / Validation Loss",
                    x_title="Epoch",
                    y_title="Loss",
                    tooltip="axis",
                    height=360,
                )

            train_result_gallery = gr.Gallery(
                label="학습 결과 시각화",
                columns=2,
                height=700,
                object_fit="contain",
            )
            with gr.Accordion("상세 학습 로그", open=False):
                train_log = gr.HTML(label="학습 로그")
            train_status_timer = gr.Timer(1.0, active=True)

            train_ui_outputs = [
                train_runtime_status,
                train_epoch_progress,
                train_quality_chart,
                train_loss_chart,
                train_result_gallery,
                train_log,
                train_start_btn,
                train_stop_btn,
            ]

            train_event = train_start_btn.click(
                fn=run_train,
                inputs=[epochs_slider, imgsz_slider, batch_slider, patience_slider, device_radio,
                        train_name, base_model_dd, training_dataset_select],
                outputs=train_ui_outputs,
            )
            train_stop_btn.click(
                fn=stop_train,
                outputs=train_ui_outputs,
            )
            base_model_refresh.click(
                fn=refresh_base_model_dropdown,
                outputs=base_model_dd,
            )

        # ── 추론 ─────────────────────────────────────────────
        with gr.Tab("추론") as panel_inference:
            gr.Markdown("### 학습된 모델로 실시간 추론")

            _inf_models = models.list_trained_models()
            with gr.Row():
                inf_model_path = gr.Dropdown(
                    choices=_inf_models,
                    value=(_inf_models[0][1] if _inf_models else None),
                    label="학습된 모델 선택 (best.pt)",
                    info="runs/detect/ 의 학습 완료 모델 — 최신 학습 순 (괄호 안은 생성 날짜). "
                         "비어 있으면 먼저 학습을 완료하세요.",
                    scale=4,
                )
                inf_model_refresh = gr.Button("모델 목록 새로고침", scale=1)
            with gr.Row():
                inf_conf = gr.Slider(
                    minimum=0.05, maximum=0.95, value=0.5, step=0.05,
                    label="신뢰도 임계값 (conf)",
                )
                inf_skip = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="추론 간격 (N프레임마다 1회)",
                    info="1=매 프레임 추론(느림), 3=3프레임마다 추론(권장), 10=빠르지만 부정확",
                )

            with gr.Row():
                inf_source_type = gr.Radio(
                    choices=["웹캠", "YouTube URL", "비디오 파일", "이미지 폴더"],
                    value="웹캠",
                    label="소스",
                )

            inf_youtube_url = gr.Textbox(
                placeholder="https://www.youtube.com/watch?v=...",
                label="YouTube URL",
                elem_id="tab5_youtube_url",
                elem_classes=["src-hidden"],
            )
            with gr.Row(elem_id="tab5_youtube_sample_row", elem_classes=["src-hidden"]):
                inf_youtube_sample_btn = gr.Button(
                    "샘플 YouTube URL 사용\nsamples/sample_url.txt",
                    elem_classes=["sample-card"],
                )
            with gr.Row(elem_id="tab5_webcam_row"):
                inf_webcam_index = gr.Dropdown(
                    choices=_webcam_choices,
                    value=_webcam_value,
                    allow_custom_value=True,
                    label="웹캠 장비",
                    info="서버 카메라가 먼저, 현재 접속 기기의 카메라가 다음에 표시됩니다.",
                    scale=4,
                )
                inf_webcam_refresh = gr.Button("서버 + 접속 기기 카메라 검색", scale=1)
            inf_webcam_browser_payload = gr.Textbox(visible=False)
            with gr.Group(
                elem_id="inference_browser_stage",
                elem_classes=["src-hidden"],
            ):
                inf_webcam_browser_input = gr.Image(
                    sources=["webcam"],
                    type="numpy",
                    streaming=True,
                    interactive=True,
                    visible=False,
                    show_label=False,
                    height=360,
                    elem_id="inference_browser_camera",
                    elem_classes=["live-preview"],
                    webcam_options=gr.WebcamOptions(mirror=False),
                )
                inf_browser_overlay = gr.HTML(
                    value="",
                    elem_id="inference_browser_overlay",
                )
            with gr.Row(elem_id="tab5_video_row", elem_classes=["src-hidden"]):
                inf_video_file = gr.File(
                    label="비디오 파일 업로드 (비우면 Tab 1 업로드 자동 사용)",
                    file_count="single",
                    type="filepath",
                    file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"],
                    height=120,
                )
            with gr.Row(elem_id="tab5_video_sample_row", elem_classes=["src-hidden"]):
                inf_video_sample_btn = gr.Button(
                    "샘플 비디오 파일 사용\nsamples/sample.mp4",
                    elem_classes=["sample-card"],
                )
            with gr.Row(elem_id="tab5_folder_row", elem_classes=["src-hidden"]) as inf_folder_row:
                inf_folder_files = gr.File(
                    label="이미지 폴더 업로드 (비우면 Tab 1 업로드 자동 사용)",
                    file_count="directory",
                    type="filepath",
                    height=200,
                )
            with gr.Row(elem_id="tab5_folder_sample_row", elem_classes=["src-hidden"]):
                inf_folder_sample_btn = gr.Button(
                    "샘플 이미지 폴더 사용\nsamples/sample_image",
                    elem_classes=["sample-card"],
                )

            inference_source_toggle_js = """(s, webcamValue) => {
                    const yt = document.getElementById('tab5_youtube_url');
                    const ys = document.getElementById('tab5_youtube_sample_row');
                    const wc = document.getElementById('tab5_webcam_row');
                    const vf = document.getElementById('tab5_video_row');
                    const vs = document.getElementById('tab5_video_sample_row');
                    const fr = document.getElementById('tab5_folder_row');
                    const fs = document.getElementById('tab5_folder_sample_row');
                    const browserWebcam = s === '웹캠'
                        && String(webcamValue || '').startsWith('browser:');
                    if (yt) yt.classList.toggle('src-hidden', s !== 'YouTube URL');
                    if (ys) ys.classList.toggle('src-hidden', s !== 'YouTube URL');
                    if (wc) wc.classList.toggle('src-hidden', s !== '웹캠');
                    if (vf) vf.classList.toggle('src-hidden', s !== '비디오 파일');
                    if (vs) vs.classList.toggle('src-hidden', s !== '비디오 파일');
                    if (fr) fr.classList.toggle('src-hidden', s !== '이미지 폴더');
                    if (fs) fs.classList.toggle('src-hidden', s !== '이미지 폴더');
                    document.querySelectorAll('#inference_browser_stage').forEach(
                        (element) => element.classList.toggle('src-hidden', !browserWebcam)
                    );
                    document.querySelectorAll('#inference_result_image').forEach(
                        (element) => element.classList.toggle('src-hidden', browserWebcam)
                    );
                }"""
            inf_source_type.change(
                fn=None,
                inputs=[inf_source_type, inf_webcam_index],
                outputs=None,
                js=inference_source_toggle_js,
            )
            inf_webcam_index.change(
                fn=None,
                inputs=[inf_source_type, inf_webcam_index],
                outputs=None,
                js=inference_source_toggle_js,
            )
            inf_webcam_index.change(
                fn=configure_browser_camera,
                inputs=[browser_session_id, inf_webcam_index, inf_source_type],
                outputs=inf_webcam_browser_input,
                show_progress="hidden",
            )
            inf_source_type.change(
                fn=configure_browser_camera,
                inputs=[browser_session_id, inf_webcam_index, inf_source_type],
                outputs=inf_webcam_browser_input,
                show_progress="hidden",
            )
            inf_webcam_browser_input.stream(
                fn=receive_browser_camera_frame,
                inputs=[browser_session_id, inf_webcam_index, inf_webcam_browser_input],
                outputs=None,
                queue=False,
                show_progress="hidden",
                stream_every=0.1,
            )
            inf_source_type.change(
                fn=_inherit_folder,
                inputs=[inf_source_type, inf_folder_files, folder_files],
                outputs=inf_folder_files,
                show_progress="hidden",
            )
            inf_source_type.change(
                fn=_inherit_video,
                inputs=[inf_source_type, inf_video_file, video_file],
                outputs=inf_video_file,
                show_progress="hidden",
            )

            with gr.Row():
                inf_start_btn = gr.Button("추론 시작", variant="primary")
                inf_stop_btn  = gr.Button("중지", variant="stop")

            inf_preview = gr.Image(
                label="추론 결과",
                type="numpy",
                streaming=True,
                elem_id="inference_result_image",
                elem_classes=["live-preview"],
            )
            inf_status  = gr.Textbox(label="상태", interactive=False)

            inf_youtube_sample_btn.click(
                fn=load_sample_youtube_url,
                outputs=[inf_youtube_url, inf_status],
            )
            inf_video_sample_btn.click(
                fn=load_sample_video_file,
                outputs=[inf_video_file, inf_status],
            )
            inf_folder_sample_btn.click(
                fn=load_sample_image_folder,
                outputs=[inf_folder_files, inf_status],
            )

            inf_event = inf_start_btn.click(
                fn=run_inference,
                inputs=[browser_session_id, inf_model_path, inf_source_type, inf_youtube_url, inf_conf, inf_skip, inf_folder_files, inf_webcam_index, inf_video_file],
                outputs=[inf_preview, inf_status, inf_browser_overlay],
                js=_AUTO_START_INFERENCE_CAMERA_JS,
            )
            inf_stop_btn.click(
                fn=inference.stop,
                outputs=inf_browser_overlay,
                cancels=[inf_event],
            )
            inf_model_refresh.click(
                fn=refresh_model_dropdown,
                outputs=inf_model_path,
            )
            inf_webcam_refresh.click(
                fn=refresh_inference_webcams,
                inputs=inf_webcam_browser_payload,
                outputs=[inf_webcam_index, inf_status],
                js=_BROWSER_CAMERA_DISCOVERY_JS,
            )


        # ── 침입 감지 ────────────────────────────────────────
        with gr.Tab("침입 감지") as panel_zone:
            gr.Markdown("### Safety Cone 자동 추적 침입 감지")
            zm_session_id = gr.State(
                value=zone_monitor.create_session,
                time_to_live=3600,
                delete_callback=zone_monitor.delete_session,
            )

            _zm_models = models.list_trained_models()
            with gr.Row():
                zm_model_path = gr.Dropdown(
                    choices=_zm_models,
                    value=(_zm_models[0][1] if _zm_models else None),
                    label="학습된 모델 선택 (best.pt)",
                    info="runs/detect/ 의 학습 완료 모델 — 최신 학습 순 (괄호 안은 생성 날짜). "
                         "비어 있으면 먼저 학습을 완료하세요.",
                    scale=4,
                )
                zm_model_refresh = gr.Button("모델 목록 새로고침", scale=1)

            zm_conf = gr.Slider(
                minimum=0.05, maximum=0.95, value=0.5, step=0.05,
                label="신뢰도 임계값 (conf)",
            )
            zm_skip = gr.State(1)

            with gr.Row():
                zm_source_type = gr.Radio(
                    choices=["웹캠", "YouTube URL", "비디오 파일", "이미지 폴더"],
                    value="웹캠",
                    label="소스",
                )

            zm_youtube_url = gr.Textbox(
                placeholder="https://www.youtube.com/watch?v=...",
                label="YouTube URL",
                elem_id="tab6_youtube_url",
                elem_classes=["src-hidden"],
            )
            with gr.Row(elem_id="tab6_youtube_sample_row", elem_classes=["src-hidden"]):
                zm_youtube_sample_btn = gr.Button(
                    "샘플 YouTube URL 사용\nsamples/sample_url.txt",
                    elem_classes=["sample-card"],
                )
            with gr.Row(elem_id="tab6_webcam_row"):
                zm_webcam_index = gr.Dropdown(
                    choices=_webcam_choices,
                    value=_webcam_value,
                    allow_custom_value=True,
                    label="웹캠 장비",
                    info="서버 카메라가 먼저, 현재 접속 기기의 카메라가 다음에 표시됩니다.",
                    scale=4,
                )
                zm_webcam_refresh = gr.Button("서버 + 접속 기기 카메라 검색", scale=1)
            zm_webcam_browser_payload = gr.Textbox(visible=False)
            zm_webcam_browser_input = gr.Image(
                label="접속 기기 카메라 입력 — 스트리밍 시작 후 아래의 스트림 시작을 누르세요",
                sources=["webcam"],
                type="numpy",
                streaming=True,
                interactive=True,
                visible=False,
                height=360,
                elem_classes=["live-preview"],
                webcam_options=gr.WebcamOptions(mirror=False),
            )
            with gr.Row(elem_id="tab6_video_row", elem_classes=["src-hidden"]):
                zm_video_file = gr.File(
                    label="비디오 파일 업로드 (비우면 Tab 1 업로드 자동 사용)",
                    file_count="single",
                    type="filepath",
                    file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"],
                    height=120,
                )
            with gr.Row(elem_id="tab6_video_sample_row", elem_classes=["src-hidden"]):
                zm_video_sample_btn = gr.Button(
                    "샘플 비디오 파일 사용\nsamples/sample.mp4",
                    elem_classes=["sample-card"],
                )
            with gr.Row(elem_id="tab6_folder_row", elem_classes=["src-hidden"]) as zm_folder_row:
                zm_folder_files = gr.File(
                    label="이미지 폴더 업로드 (비우면 Tab 1 업로드 자동 사용)",
                    file_count="directory",
                    type="filepath",
                    height=200,
                )
            with gr.Row(elem_id="tab6_folder_sample_row", elem_classes=["src-hidden"]):
                zm_folder_sample_btn = gr.Button(
                    "샘플 이미지 폴더 사용\nsamples/sample_image",
                    elem_classes=["sample-card"],
                )

            zm_source_type.change(
                fn=None,
                inputs=zm_source_type,
                outputs=None,
                js="""(s) => {
                    const yt = document.getElementById('tab6_youtube_url');
                    const ys = document.getElementById('tab6_youtube_sample_row');
                    const wc = document.getElementById('tab6_webcam_row');
                    const vf = document.getElementById('tab6_video_row');
                    const vs = document.getElementById('tab6_video_sample_row');
                    const fr = document.getElementById('tab6_folder_row');
                    const fs = document.getElementById('tab6_folder_sample_row');
                    if (yt) yt.classList.toggle('src-hidden', s !== 'YouTube URL');
                    if (ys) ys.classList.toggle('src-hidden', s !== 'YouTube URL');
                    if (wc) wc.classList.toggle('src-hidden', s !== '웹캠');
                    if (vf) vf.classList.toggle('src-hidden', s !== '비디오 파일');
                    if (vs) vs.classList.toggle('src-hidden', s !== '비디오 파일');
                    if (fr) fr.classList.toggle('src-hidden', s !== '이미지 폴더');
                    if (fs) fs.classList.toggle('src-hidden', s !== '이미지 폴더');
                }""",
            )
            zm_webcam_index.change(
                fn=configure_browser_camera,
                inputs=[browser_session_id, zm_webcam_index, zm_source_type],
                outputs=zm_webcam_browser_input,
                show_progress="hidden",
            )
            zm_source_type.change(
                fn=configure_browser_camera,
                inputs=[browser_session_id, zm_webcam_index, zm_source_type],
                outputs=zm_webcam_browser_input,
                show_progress="hidden",
            )
            zm_webcam_browser_input.stream(
                fn=receive_browser_camera_frame,
                inputs=[browser_session_id, zm_webcam_index, zm_webcam_browser_input],
                outputs=None,
                queue=False,
                show_progress="hidden",
                stream_every=0.1,
            )
            zm_source_type.change(
                fn=_inherit_folder,
                inputs=[zm_source_type, zm_folder_files, folder_files],
                outputs=zm_folder_files,
                show_progress="hidden",
            )
            zm_source_type.change(
                fn=_inherit_video,
                inputs=[zm_source_type, zm_video_file, video_file],
                outputs=zm_video_file,
                show_progress="hidden",
            )

            with gr.Row():
                zm_start_btn = gr.Button("스트림 시작", variant="primary")
                zm_stop_btn  = gr.Button("중지 / 초기화", variant="stop")

            zm_preview = gr.Image(
                label="Safety Cone 자동 추적 실시간 영상",
                type="numpy",
                streaming=True,
                elem_classes=["live-preview"],
            )
            zm_stream_status = gr.Textbox(label="상태", interactive=False)
            gr.Markdown(
                "`스트림 시작`을 누르면 `Safety Cone`을 매 프레임 추적합니다. "
                "라바콘 Track이 3개 이상 잡히면 바닥 중심점의 외곽선을 연결해 "
                "침입 감지 영역을 자동 생성하고 계속 갱신합니다."
            )
            gr.Markdown(
                "**표시 색상 안내**\n\n"
                "- **초록색 영역**: 영역 안에 침입 객체가 없습니다.\n"
                "- **빨간색 영역·반투명 채우기**: 영역 안에 침입 객체가 있습니다. "
                "영역 이름의 `(N)`은 침입 객체 수입니다.\n"
                "- **주황색 영역·앵커**: 추적하던 Safety Cone의 Track ID가 현재 프레임에서 "
                "하나 이상 유실되어 마지막 위치를 유지 중입니다.\n"
                "- **하늘색 점·선**: 정상 추적 중인 Safety Cone 앵커와 자동 감시 경계입니다."
            )

            gr.Markdown("#### 수동 다각형 영역 추가")
            gr.Markdown(
                "스트림 실행 중 **현재 프레임 가져오기**를 누른 뒤, "
                "영역 꼭짓점을 순서대로 3개 이상 클릭하고 **수동 다각형 완료**를 누르세요."
            )
            zm_manual_mode = gr.State(zone_monitor.MODE_MANUAL)
            with gr.Row():
                zm_snapshot_btn = gr.Button(
                    "현재 프레임 가져오기",
                    variant="secondary",
                )
                zm_zone_label = gr.Textbox(
                    label="수동 영역 이름",
                    placeholder="예: Entrance",
                )
            zm_zone_editor = gr.Image(
                label="수동 영역 편집 (고정 프레임의 꼭짓점을 클릭하세요)",
                type="numpy",
                interactive=False,
            )
            with gr.Row():
                zm_undo_btn = gr.Button("마지막 점 취소")
                zm_clear_draft_btn = gr.Button("작성 중인 점 지우기")
                zm_finish_btn = gr.Button("수동 다각형 완료", variant="primary")
            zm_zone_status = gr.Textbox(
                label="수동 영역 편집 상태",
                interactive=False,
            )

            zm_youtube_sample_btn.click(
                fn=load_sample_youtube_url,
                outputs=[zm_youtube_url, zm_stream_status],
            )
            zm_video_sample_btn.click(
                fn=load_sample_video_file,
                outputs=[zm_video_file, zm_stream_status],
            )
            zm_folder_sample_btn.click(
                fn=load_sample_image_folder,
                outputs=[zm_folder_files, zm_stream_status],
            )

            zm_prepare_event = zm_start_btn.click(
                fn=zone_monitor.prepare_stream,
                inputs=zm_session_id,
                outputs=[
                    zm_preview,
                    zm_zone_editor,
                    zm_stream_status,
                    zm_zone_status,
                ],
                show_progress="hidden",
            )
            zm_stream_event = zm_prepare_event.then(
                fn=run_zone_stream,
                inputs=[browser_session_id, zm_session_id, zm_source_type, zm_youtube_url, zm_model_path, zm_conf, zm_skip, zm_folder_files, zm_webcam_index, zm_video_file],
                outputs=[zm_preview, zm_stream_status],
                concurrency_id="zone_stream",
                concurrency_limit=1,
            )
            zm_stop_btn.click(
                fn=zone_monitor.reset,
                inputs=zm_session_id,
                outputs=[
                    zm_preview,
                    zm_zone_editor,
                    zm_stream_status,
                    zm_zone_status,
                ],
                cancels=[zm_stream_event],
            )
            zm_snapshot_btn.click(
                fn=zone_monitor.capture_editor_frame,
                inputs=[zm_session_id, zm_manual_mode],
                outputs=[zm_zone_editor, zm_zone_status],
                concurrency_id="zone_editor",
                concurrency_limit=1,
            )
            zm_zone_editor.select(
                fn=on_zone_editor_select,
                inputs=[zm_session_id, zm_manual_mode],
                outputs=[zm_zone_editor, zm_zone_status],
                concurrency_id="zone_editor",
                concurrency_limit=1,
            )
            zm_undo_btn.click(
                fn=zone_monitor.undo_draft_point,
                inputs=[zm_session_id, zm_manual_mode],
                outputs=[zm_zone_editor, zm_zone_status],
                concurrency_id="zone_editor",
                concurrency_limit=1,
            )
            zm_clear_draft_btn.click(
                fn=zone_monitor.clear_draft,
                inputs=zm_session_id,
                outputs=[zm_zone_editor, zm_zone_status],
                concurrency_id="zone_editor",
                concurrency_limit=1,
            )
            zm_finish_btn.click(
                fn=zone_monitor.finish_draft,
                inputs=[zm_session_id, zm_manual_mode, zm_zone_label],
                outputs=[zm_zone_editor, zm_zone_status],
                concurrency_id="zone_editor",
                concurrency_limit=1,
            )
            zm_model_refresh.click(
                fn=refresh_model_dropdown,
                outputs=zm_model_path,
            )
            zm_webcam_refresh.click(
                fn=refresh_zone_webcams,
                inputs=zm_webcam_browser_payload,
                outputs=[zm_webcam_index, zm_stream_status],
                js=_BROWSER_CAMERA_DISCOVERY_JS,
            )

    dataset_ui_outputs = [
        dataset_registry_state,
        training_dataset_select,
        dataset_registry_table,
        dataset_import_status,
        dataset_selection_status,
        train_dataset_summary,
    ]
    archive_dataset_add.click(
        add_archive_datasets,
        inputs=[
            dataset_archives,
            dataset_registry_state,
            training_dataset_select,
        ],
        outputs=dataset_ui_outputs,
    )
    roboflow_dataset_add.click(
        add_roboflow_dataset,
        inputs=[
            roboflow_universe_url,
            dataset_registry_state,
            training_dataset_select,
        ],
        outputs=dataset_ui_outputs + [roboflow_downloaded_select],
        show_progress="full",
    )
    roboflow_universe_url.submit(
        add_roboflow_dataset,
        inputs=[
            roboflow_universe_url,
            dataset_registry_state,
            training_dataset_select,
        ],
        outputs=dataset_ui_outputs + [roboflow_downloaded_select],
        show_progress="full",
    )
    roboflow_downloaded_select.input(
        register_downloaded_roboflow,
        inputs=[
            roboflow_downloaded_select,
            dataset_registry_state,
            training_dataset_select,
        ],
        outputs=dataset_ui_outputs + [
            roboflow_downloaded_select,
            roboflow_register_loading,
        ],
        show_progress="full",
        concurrency_id="roboflow_register",
        concurrency_limit=1,
    )
    roboflow_downloaded_refresh.click(
        refresh_downloaded_roboflow,
        outputs=[roboflow_downloaded_select, dataset_import_status],
    )
    dataset_refresh_btn.click(
        refresh_datasets,
        inputs=[dataset_registry_state, training_dataset_select],
        outputs=dataset_ui_outputs,
    )
    dataset_remove_btn.click(
        remove_external_datasets,
        inputs=[dataset_registry_state, training_dataset_select],
        outputs=dataset_ui_outputs,
    )
    training_dataset_select.change(
        update_dataset_selection_status,
        inputs=[dataset_registry_state, training_dataset_select],
        outputs=[dataset_selection_status, train_dataset_summary],
    )
    panel_dataset_import.select(
        refresh_datasets,
        inputs=[dataset_registry_state, training_dataset_select],
        outputs=dataset_ui_outputs,
    )

    # 데이터셋 단계 진입 시 클래스 목록 로드 (네이티브 탭 select 이벤트).
    # Tab 2 프롬프트가 직전 로드와 다르거나(재라벨링) 아직 로드된 적 없으면 새로
    # 로드하고, 그 외 단순 탭 전환에는 그대로 두어 사용자가 편집한 이름을 보존한다.
    def _maybe_load_classes(label_prompts, last_label, class_state):
        if class_state and (label_prompts or "") == (last_label or ""):
            return gr.update(), gr.update(), gr.update()
        return load_classes(label_prompts)

    panel_dataset.select(
        _maybe_load_classes,
        inputs=[prompts_input, ds_last_label, ds_class_state],
        outputs=[ds_class_state, ds_prompts, ds_last_label],
    )
    panel_labeler.select(
        refresh_label_source_choices,
        inputs=label_source_select,
        outputs=label_source_select,
    )
    panel_dataset.select(
        refresh_validation_source_choices,
        inputs=validation_source_select,
        outputs=validation_source_select,
    )

    # 추론·침입 감지 탭 진입 시 학습된 모델 목록 자동 새로고침
    # (방금 학습을 끝낸 모델이 바로 목록에 반영되도록)
    panel_inference.select(refresh_model_dropdown, outputs=inf_model_path)
    panel_zone.select(refresh_model_dropdown, outputs=zm_model_path)
    # 학습 탭 진입 시 베이스 모델 목록도 자동 새로고침
    panel_train.select(refresh_base_model_dropdown, outputs=base_model_dd)
    panel_train.select(training_panel_update, outputs=train_ui_outputs)
    train_status_timer.tick(
        training_panel_update,
        outputs=train_ui_outputs,
        show_progress="hidden",
    )
    demo.load(
        training_panel_update,
        outputs=train_ui_outputs,
        show_progress="hidden",
    )


if __name__ == "__main__":
    models.ensure_models()
    demo.queue().launch(
        theme=gr.themes.Default(),
        css=_CSS,
        allowed_paths=[str(SAMPLES_DIR.resolve())],
    )
