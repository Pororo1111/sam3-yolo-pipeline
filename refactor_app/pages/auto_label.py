import gradio as gr

from pipeline import labeler


def render():
    gr.Markdown("## 2. SAM3 오토라벨링")
    gr.Markdown("프롬프트로 샘플을 먼저 확인한 뒤 전체 프레임에 라벨을 저장합니다.")

    prompts = gr.Textbox(label="클래스 프롬프트", placeholder="person, car")
    with gr.Row():
        conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
        n_preview = gr.Slider(1, 16, value=4, step=1, label="미리보기 샘플 수")

    with gr.Row():
        preview_btn = gr.Button("미리보기", variant="secondary")
        label_btn = gr.Button("전체 라벨링 시작", variant="primary")
        stop_btn = gr.Button("중지", variant="stop")

    gallery = gr.Gallery(label="샘플 미리보기", columns=4, height=360)
    preview = gr.Image(label="진행 미리보기", type="numpy", streaming=True)
    status = gr.Textbox(label="상태", interactive=False)

    preview_event = preview_btn.click(
        labeler.preview,
        inputs=[prompts, conf, n_preview],
        outputs=[gallery, status],
    )
    label_event = label_btn.click(
        labeler.label,
        inputs=[prompts, conf],
        outputs=[preview, status],
    )
    stop_btn.click(labeler.stop, cancels=[preview_event, label_event])

    return prompts
