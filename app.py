import html as _html
from pathlib import Path

import gradio as gr
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


def run_capture(source_type, youtube_url, capture_fps, folder_files, webcam_index, video_file):
    yield from extractor.capture(
        source_type,
        youtube_url,
        int(capture_fps),
        folder_files,
        webcam_index,
        video_file,
    )


def prepare_capture_preview():
    """공용 이미지 스트림을 비우고 새 캡처를 준비한다."""
    return None, "실시간 미리보기를 준비합니다."


def stop_capture_and_reset():
    """캡처 중지 후 미리보기 영역을 초기 상태로 되돌린다."""
    status = extractor.stop()
    return None, status


def run_inference(model_path, source_type, youtube_url, conf, infer_every, folder_files, webcam_index, video_file):
    yield from inference.predict(model_path, source_type, youtube_url, conf, infer_every, folder_files, webcam_index, video_file)


def run_zone_stream(
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


def refresh_webcam_dropdown():
    """웹캠 장비 목록을 다시 감지해 드롭다운 choices/value 를 갱신한다."""
    choices, value = webcams.refresh_webcam_dropdown()
    return gr.update(choices=choices, value=value)



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
    )
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
    yield _dataset_ui_result(records or [], selected or [], message)


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


def run_label(prompts_str, conf):
    yield from labeler.label(prompts_str, float(conf))


def on_gallery_select(prompts_str, filter_empty, evt: gr.SelectData):
    """갤러리 선택 이벤트 래퍼 — SelectData 어노테이션으로 evt 주입을 명시."""
    return dataset.select_frame(prompts_str, filter_empty, evt)


def on_zone_editor_select(session_id, mode, evt: gr.SelectData):
    """고정 편집 프레임의 자연 크기 좌표를 영역 서비스에 전달한다."""
    return zone_monitor.select_editor_point(session_id, mode, evt.index)


def run_label_preview(prompts_str, conf, n_preview):
    yield from labeler.preview(prompts_str, float(conf), int(n_preview))


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
    lr0,
    device,
    name,
    base_model,
    dataset_yamls=None,
):
    for text in trainer.train(
        epochs,
        imgsz,
        batch,
        lr0,
        device,
        name,
        base_model,
        dataset_yamls,
    ):
        escaped = _html.escape(text)
        yield f'<div style="{_WRAP_STYLE}"><pre style="{_PRE_STYLE}">{escaped}</pre></div>'


_CSS = (
    ".src-hidden { display: none !important; }"
    ".cls-name-box input { font-weight:600; }"
    # 본문 패널이 좁은 화면에서도 넘치지 않도록
    ".main-panel { max-width: 1100px; margin: 0 auto; width: 100%; }"
    # 미리보기 갤러리: 항상 4열 격자로 고정. Gradio는 이미지 수에 맞춰 --grid-cols를
    # 줄여(1장이면 1열) 이미지가 칸 전체로 커지므로, grid-template-columns를 4열로 강제.
    "#label_gallery .grid-container { grid-template-columns: repeat(4, minmax(0, 1fr)) !important; }"
    ".sample-card button { min-height: 72px; border: 1px solid #d4d4d8 !important; "
    "background: #fff !important; color: #18181b !important; text-align: left; "
    "justify-content: flex-start; box-shadow: 0 1px 2px rgba(0,0,0,.06); }"
)

_webcam_choices, _webcam_value = webcams.refresh_webcam_dropdown()


with gr.Blocks(title="YOLO 파이프라인") as demo:

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
                placeholder="https://www.youtube.com/watch?v=...",
                label="YouTube URL",
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
                    label="웹캠 장비",
                    info="감지된 카메라를 선택하세요. 목록이 비어 있으면 새로고침을 눌러 재탐색합니다.",
                    scale=4,
                )
                webcam_refresh = gr.Button("웹캠 목록 새로고침", scale=1)
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
            with gr.Row():
                cap_start_btn = gr.Button("캡처 시작", variant="primary")
                cap_stop_btn  = gr.Button("중지", variant="stop")

            cap_preview = gr.Image(label="실시간 미리보기", type="numpy", streaming=True)
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
                inputs=[source_type, youtube_url, capture_fps, folder_files, webcam_index, video_file],
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
                fn=refresh_webcam_dropdown,
                outputs=webcam_index,
            )

        # ── SAM3 오토라벨링 ──────────────────────────────────
        with gr.Tab("SAM3 오토라벨링"):
            gr.Markdown("### SAM3 텍스트 프롬프트로 자동 라벨링")
            gr.Markdown(
                "프롬프트와 conf를 입력하고 **① 미리보기**로 샘플 라벨을 확인하세요. "
                "결과가 괜찮으면 **② 전체 라벨링 시작**으로 모든 프레임에 적용합니다."
            )

            prompts_input = gr.Textbox(
                placeholder="person, car, bicycle",
                label="클래스 프롬프트 (쉼표 구분)",
                value="",
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
            )
            label_status  = gr.Textbox(label="상태", interactive=False)

            preview_event = label_preview_btn.click(
                fn=run_label_preview,
                inputs=[prompts_input, conf_slider, n_preview_slider],
                outputs=[label_gallery, label_status],
            )
            label_event = label_start_btn.click(
                fn=run_label,
                inputs=[prompts_input, conf_slider],
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
                "**이름은 이 칸에서 직접 수정**하면 데이터셋 구성·미리보기에 그대로 반영됩니다."
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
                    value=True,
                )
                val_ratio_slider = gr.Slider(
                    minimum=0.1, maximum=0.4, value=0.2, step=0.05,
                    label="Validation 비율",
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
                inputs=[ds_prompts, val_ratio_slider, filter_empty_chk],
                outputs=build_status,
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
                lr0_slider = gr.Slider(
                    minimum=0.0001, maximum=0.1, value=0.01, step=0.0001,
                    label="Learning Rate (lr0)",
                    info="초기 학습률. ultralytics 기본값 0.01. "
                         "학습이 불안정하거나 loss가 튀면 0.001~0.005로 낮추세요",
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
                train_stop_btn  = gr.Button("중지", variant="stop")

            train_log = gr.HTML(label="학습 로그")

            train_event = train_start_btn.click(
                fn=run_train,
                inputs=[epochs_slider, imgsz_slider, batch_slider, lr0_slider, device_radio,
                        train_name, base_model_dd, training_dataset_select],
                outputs=train_log,
            )
            train_stop_btn.click(
                fn=trainer.stop,
                cancels=[train_event],
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
                    minimum=0.05, maximum=0.95, value=0.15, step=0.05,
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
                    label="웹캠 장비",
                    info="추론에 사용할 카메라를 선택하세요.",
                    scale=4,
                )
                inf_webcam_refresh = gr.Button("웹캠 목록 새로고침", scale=1)
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

            inf_source_type.change(
                fn=None,
                inputs=inf_source_type,
                outputs=None,
                js="""(s) => {
                    const yt = document.getElementById('tab5_youtube_url');
                    const ys = document.getElementById('tab5_youtube_sample_row');
                    const wc = document.getElementById('tab5_webcam_row');
                    const vf = document.getElementById('tab5_video_row');
                    const vs = document.getElementById('tab5_video_sample_row');
                    const fr = document.getElementById('tab5_folder_row');
                    const fs = document.getElementById('tab5_folder_sample_row');
                    if (yt) yt.classList.toggle('src-hidden', s !== 'YouTube URL');
                    if (ys) ys.classList.toggle('src-hidden', s !== 'YouTube URL');
                    if (wc) wc.classList.toggle('src-hidden', s !== '웹캠');
                    if (vf) vf.classList.toggle('src-hidden', s !== '비디오 파일');
                    if (vs) vs.classList.toggle('src-hidden', s !== '비디오 파일');
                    if (fr) fr.classList.toggle('src-hidden', s !== '이미지 폴더');
                    if (fs) fs.classList.toggle('src-hidden', s !== '이미지 폴더');
                }""",
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

            inf_preview = gr.Image(label="추론 결과", type="numpy", streaming=True)
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
                inputs=[inf_model_path, inf_source_type, inf_youtube_url, inf_conf, inf_skip, inf_folder_files, inf_webcam_index, inf_video_file],
                outputs=[inf_preview, inf_status],
            )
            inf_stop_btn.click(
                fn=inference.stop,
                cancels=[inf_event],
            )
            inf_model_refresh.click(
                fn=refresh_model_dropdown,
                outputs=inf_model_path,
            )
            inf_webcam_refresh.click(
                fn=refresh_webcam_dropdown,
                outputs=inf_webcam_index,
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
                minimum=0.05, maximum=0.95, value=0.15, step=0.05,
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
                    label="웹캠 장비",
                    info="침입 감시에 사용할 카메라를 선택하세요.",
                    scale=4,
                )
                zm_webcam_refresh = gr.Button("웹캠 목록 새로고침", scale=1)
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
                inputs=[zm_session_id, zm_source_type, zm_youtube_url, zm_model_path, zm_conf, zm_skip, zm_folder_files, zm_webcam_index, zm_video_file],
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
                fn=refresh_webcam_dropdown,
                outputs=zm_webcam_index,
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
        outputs=dataset_ui_outputs,
    )
    roboflow_universe_url.submit(
        add_roboflow_dataset,
        inputs=[
            roboflow_universe_url,
            dataset_registry_state,
            training_dataset_select,
        ],
        outputs=dataset_ui_outputs,
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

    # 추론·침입 감지 탭 진입 시 학습된 모델 목록 자동 새로고침
    # (방금 학습을 끝낸 모델이 바로 목록에 반영되도록)
    panel_inference.select(refresh_model_dropdown, outputs=inf_model_path)
    panel_zone.select(refresh_model_dropdown, outputs=zm_model_path)
    # 학습 탭 진입 시 베이스 모델 목록도 자동 새로고침
    panel_train.select(refresh_base_model_dropdown, outputs=base_model_dd)


if __name__ == "__main__":
    models.ensure_models()
    demo.queue().launch(
        theme=gr.themes.Default(),
        css=_CSS,
        allowed_paths=[str(SAMPLES_DIR.resolve())],
    )
