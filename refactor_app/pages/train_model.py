import gradio as gr

from pipeline import models, trainer


def _base_choices():
    return [("처음부터", "")] + models.list_trained_models()


def refresh_base_models():
    choices = _base_choices()
    return gr.update(choices=choices, value=choices[0][1] if choices else "")


def run_train(epochs, imgsz, batch, lr0, device, name, base_model):
    yield from trainer.train(int(epochs), int(imgsz), int(batch), float(lr0), device, name, base_model)


def render():
    gr.Markdown("## 4. YOLO 학습")
    gr.Markdown("구성된 데이터셋으로 학습하고 결과 모델을 새 run 폴더에 저장합니다.")

    with gr.Row():
        base_model = gr.Dropdown(choices=_base_choices(), value="", label="베이스 모델")
        refresh_btn = gr.Button("모델 목록 새로고침")

    with gr.Row():
        epochs = gr.Slider(1, 300, value=50, step=1, label="Epochs")
        imgsz = gr.Slider(320, 1280, value=640, step=32, label="Image size")
        batch = gr.Slider(1, 64, value=16, step=1, label="Batch")

    with gr.Row():
        lr0 = gr.Number(value=0.01, label="lr0")
        device = gr.Radio(["auto", "cpu", "0"], value="auto", label="Device")
        name = gr.Textbox(value="train", label="결과 이름")

    with gr.Row():
        start_btn = gr.Button("학습 시작", variant="primary")
        stop_btn = gr.Button("중지", variant="stop")

    log = gr.Textbox(label="학습 로그", lines=20, interactive=False)

    refresh_btn.click(refresh_base_models, outputs=base_model)
    train_event = start_btn.click(
        run_train,
        inputs=[epochs, imgsz, batch, lr0, device, name, base_model],
        outputs=log,
    )
    stop_btn.click(trainer.stop, cancels=[train_event])
