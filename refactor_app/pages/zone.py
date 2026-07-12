import gradio as gr

from pipeline import models, webcams, zone_monitor
from refactor_app.pages import source_capture

VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"]
SOURCE_TYPES = ["YouTube URL", "웹캠", "비디오 파일", "이미지 폴더"]


def _visible_for(source_type: str):
    return (
        gr.update(visible=source_type == "YouTube URL"),
        gr.update(visible=source_type == "웹캠"),
        gr.update(visible=source_type == "비디오 파일"),
        gr.update(visible=source_type == "이미지 폴더"),
    )


def _model_choices():
    return models.list_trained_models()


def refresh_models():
    choices = _model_choices()
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def refresh_webcams():
    choices, value = webcams.list_webcam_choices()
    return gr.update(choices=choices, value=value)


def run_stream(source_type, youtube_url, model_path, conf, infer_every, folder_files, webcam_index, video_file):
    yield from zone_monitor.stream(source_type, youtube_url, model_path, float(conf), int(infer_every), folder_files, webcam_index, video_file)


def stop_stream():
    zone_monitor.reset()
    return None, "중지됨 — 영역과 마지막 프레임을 초기화했습니다."


def render():
    gr.Markdown("## 6. 침입 감지")
    gr.Markdown("영상·웹캠은 현재 프레임 위에 박스와 영역을 그리고, 이미지 폴더는 감시 결과를 슬라이드쇼처럼 표시합니다.")

    model_choices = _model_choices()
    model_path = gr.Dropdown(
        choices=model_choices,
        value=model_choices[0][1] if model_choices else None,
        label="학습된 모델",
    )
    refresh_model_btn = gr.Button("모델 목록 새로고침")

    source_type = gr.Radio(SOURCE_TYPES, value="웹캠", label="소스")
    with gr.Row():
        conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
        infer_every = gr.Slider(1, 30, value=3, step=1, label="추론 간격")

    with gr.Group(visible=False) as youtube_group:
        youtube_url = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        youtube_sample_btn = gr.Button("샘플 YouTube URL 사용")

    webcam_choices, webcam_value = webcams.list_webcam_choices()
    with gr.Group(visible=True) as webcam_group:
        with gr.Row():
            webcam_index = gr.Dropdown(choices=webcam_choices, value=webcam_value, label="웹캠 장비", scale=4)
            webcam_refresh_btn = gr.Button("웹캠 목록 새로고침", scale=1)

    with gr.Group(visible=False) as video_group:
        video_file = gr.File(label="비디오 파일 업로드", file_count="single", type="filepath", file_types=VIDEO_EXTS)
        video_sample_btn = gr.Button("샘플 비디오 파일 사용")

    with gr.Group(visible=False) as folder_group:
        folder_files = gr.File(label="이미지 폴더 업로드", file_count="directory", type="filepath")
        folder_sample_btn = gr.Button("샘플 이미지 폴더 사용")

    with gr.Row():
        start_btn = gr.Button("감시 시작", variant="primary")
        stop_btn = gr.Button("중지", variant="stop")

    preview = gr.Image(label="감시 미리보기", type="numpy", streaming=True)
    status = gr.Textbox(label="상태", interactive=False)

    with gr.Row():
        zone_prompt = gr.Textbox(label="감시 영역 설명", placeholder="화면 왼쪽 출입구")
        ollama_model = gr.Textbox(value="gemma4:e4b", label="Ollama 모델")
    set_zone_btn = gr.Button("영역 설정", variant="secondary")
    zone_status = gr.Textbox(label="영역 설정 상태", interactive=False)
    llm_response = gr.Code(label="LLM 응답", language="json")

    refresh_model_btn.click(refresh_models, outputs=model_path)
    source_type.change(_visible_for, inputs=source_type, outputs=[youtube_group, webcam_group, video_group, folder_group])
    youtube_sample_btn.click(source_capture.load_sample_youtube_url, outputs=[youtube_url, status])
    video_sample_btn.click(source_capture.load_sample_video_file, outputs=[video_file, status])
    folder_sample_btn.click(source_capture.load_sample_image_folder, outputs=[folder_files, status])
    webcam_refresh_btn.click(refresh_webcams, outputs=webcam_index)

    stream_event = start_btn.click(
        run_stream,
        inputs=[source_type, youtube_url, model_path, conf, infer_every, folder_files, webcam_index, video_file],
        outputs=[preview, status],
    )
    stop_btn.click(stop_stream, outputs=[preview, status], cancels=[stream_event])
    set_zone_btn.click(zone_monitor.set_zone, inputs=[zone_prompt, ollama_model], outputs=[zone_status, llm_response])
