import gradio as gr

from refactor_app.pages import auto_label, dataset_review, infer, source_capture, train_model, zone


def build_demo():
    with gr.Blocks(title="YOLO 파이프라인 리팩토링") as demo:
        gr.Markdown("# YOLO 파이프라인 리팩토링")
        gr.Markdown("기존 앱을 건드리지 않고 페이지 단위로 단순하게 다시 구성하는 리팩토링 버전입니다.")

        with gr.Tab("1. 소스 선택 & 프레임 추출"):
            source_capture.render()
        with gr.Tab("2. SAM3 오토라벨링"):
            label_prompts = auto_label.render()
        with gr.Tab("3. 데이터셋 검토 & 구성"):
            dataset_review.render(label_prompts)
        with gr.Tab("4. YOLO 학습"):
            train_model.render()
        with gr.Tab("5. 추론"):
            infer.render()
        with gr.Tab("6. 침입 감지"):
            zone.render()
    return demo


if __name__ == "__main__":
    build_demo().launch()
