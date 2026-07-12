import gradio as gr

from pipeline import inference, models, webcams
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


def run_predict(model_path, source_type, youtube_url, conf, infer_every, folder_files, webcam_index, video_file):
    yield from inference.predict(model_path, source_type, youtube_url, float(conf), int(infer_every), folder_files, webcam_index, video_file)


def render():
    gr.Markdown("## 5. 추론")
    gr.Markdown("영상·웹캠은 들어오는 프레임 위에 추론 결과를 그려 보여주고, 이미지 폴더는 결과를 슬라이드쇼처럼 표시합니다.")

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
        start_btn = gr.Button("추론 시작", variant="primary")
        stop_btn = gr.Button("중지", variant="stop")

    preview = gr.Image(label="추론 미리보기", type="numpy", streaming=True)
    status = gr.Textbox(label="상태", interactive=False)

    refresh_model_btn.click(refresh_models, outputs=model_path)
    source_type.change(_visible_for, inputs=source_type, outputs=[youtube_group, webcam_group, video_group, folder_group])
    youtube_sample_btn.click(source_capture.load_sample_youtube_url, outputs=[youtube_url, status])
    video_sample_btn.click(source_capture.load_sample_video_file, outputs=[video_file, status])
    folder_sample_btn.click(source_capture.load_sample_image_folder, outputs=[folder_files, status])
    webcam_refresh_btn.click(refresh_webcams, outputs=webcam_index)

    predict_event = start_btn.click(
        run_predict,
        inputs=[model_path, source_type, youtube_url, conf, infer_every, folder_files, webcam_index, video_file],
        outputs=[preview, status],
    )
    stop_btn.click(inference.stop, cancels=[predict_event])
