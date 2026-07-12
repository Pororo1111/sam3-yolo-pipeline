from pathlib import Path

import gradio as gr

from pipeline import extractor, webcams

SAMPLES_DIR = Path("samples")
SAMPLE_URL_PATH = SAMPLES_DIR / "sample_url.txt"
SAMPLE_VIDEO_PATH = SAMPLES_DIR / "sample.mp4"
SAMPLE_IMAGE_DIR = SAMPLES_DIR / "sample_image"
SAMPLE_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"]
SOURCE_TYPES = ["YouTube URL", "웹캠", "비디오 파일", "이미지 폴더"]


def _visible_for(source_type: str):
    return (
        gr.update(visible=source_type == "YouTube URL"),
        gr.update(visible=source_type == "웹캠"),
        gr.update(visible=source_type == "비디오 파일"),
        gr.update(visible=source_type == "이미지 폴더"),
    )


def _webcam_choices():
    choices, value = webcams.list_webcam_choices()
    return choices, value


def refresh_webcams():
    choices, value = _webcam_choices()
    return gr.update(choices=choices, value=value)


def load_sample_youtube_url():
    if not SAMPLE_URL_PATH.exists():
        return gr.update(), f"샘플 URL 파일을 찾을 수 없습니다: {SAMPLE_URL_PATH}"
    url = SAMPLE_URL_PATH.read_text(encoding="utf-8", errors="ignore").strip()
    if not url:
        return gr.update(), f"샘플 URL 파일이 비어 있습니다: {SAMPLE_URL_PATH}"
    return gr.update(value=url), "샘플 YouTube URL을 불러왔습니다."


def load_sample_video_file():
    if not SAMPLE_VIDEO_PATH.exists():
        return gr.update(), f"샘플 비디오 파일을 찾을 수 없습니다: {SAMPLE_VIDEO_PATH}"
    return gr.update(value=str(SAMPLE_VIDEO_PATH)), "샘플 비디오 파일을 선택했습니다."


def load_sample_image_folder():
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
    for frame, status in extractor.capture(
        source_type=source_type,
        youtube_url=youtube_url,
        capture_fps=int(capture_fps),
        folder_files=folder_files,
        webcam_index=webcam_index,
        video_file=video_file,
        emit_preview=True,
    ):
        yield frame, status


def stop_capture():
    return None, extractor.stop()


def render():
    gr.Markdown("## 1. 소스 선택 & 프레임 추출")
    gr.Markdown("소스를 고르고 **캡처 시작**을 누르면 같은 미리보기 영역에서 진행 화면과 상태가 갱신됩니다.")

    webcam_choices, webcam_value = _webcam_choices()

    source_type = gr.Radio(
        SOURCE_TYPES,
        value="YouTube URL",
        label="소스",
    )
    capture_fps = gr.Slider(
        minimum=1,
        maximum=30,
        value=5,
        step=1,
        label="캡처 FPS",
        info="이미지 폴더에서는 사용하지 않습니다.",
    )

    with gr.Group(visible=True) as youtube_group:
        youtube_url = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        youtube_sample_btn = gr.Button("샘플 YouTube URL 사용")

    with gr.Group(visible=False) as webcam_group:
        with gr.Row():
            webcam_index = gr.Dropdown(
                choices=webcam_choices,
                value=webcam_value,
                label="웹캠 장비",
                scale=4,
            )
            webcam_refresh_btn = gr.Button("웹캠 목록 새로고침", scale=1)

    with gr.Group(visible=False) as video_group:
        video_file = gr.File(
            label="비디오 파일 업로드",
            file_count="single",
            type="filepath",
            file_types=VIDEO_EXTS,
        )
        video_sample_btn = gr.Button("샘플 비디오 파일 사용")

    with gr.Group(visible=False) as folder_group:
        folder_files = gr.File(
            label="이미지 폴더 업로드",
            file_count="directory",
            type="filepath",
        )
        folder_sample_btn = gr.Button("샘플 이미지 폴더 사용")

    with gr.Row():
        start_btn = gr.Button("캡처 시작", variant="primary")
        stop_btn = gr.Button("중지", variant="stop")

    preview = gr.Image(label="진행 미리보기", type="numpy", streaming=True)
    status = gr.Textbox(label="상태", interactive=False)

    source_type.change(
        _visible_for,
        inputs=source_type,
        outputs=[youtube_group, webcam_group, video_group, folder_group],
    )
    youtube_sample_btn.click(load_sample_youtube_url, outputs=[youtube_url, status])
    video_sample_btn.click(load_sample_video_file, outputs=[video_file, status])
    folder_sample_btn.click(load_sample_image_folder, outputs=[folder_files, status])
    webcam_refresh_btn.click(refresh_webcams, outputs=webcam_index)

    capture_event = start_btn.click(
        run_capture,
        inputs=[source_type, youtube_url, capture_fps, folder_files, webcam_index, video_file],
        outputs=[preview, status],
    )
    stop_btn.click(stop_capture, outputs=[preview, status], cancels=[capture_event])
