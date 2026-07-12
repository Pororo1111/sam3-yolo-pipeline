import gradio as gr

from pipeline import dataset


def _class_summary(label_prompts: str):
    classes = dataset.scan_classes(label_prompts)
    if not classes:
        return "클래스가 없습니다. 먼저 라벨링을 실행하세요."
    return "\n".join(f"- ID {c['id']}: {c['name']} · {c['count']}개 프레임" for c in classes)


def _select_frame(prompts, filter_empty, evt: gr.SelectData):
    return dataset.select_frame(prompts, filter_empty, evt)


def render(label_prompts=None):
    gr.Markdown("## 3. 데이터셋 검토 & 구성")
    gr.Markdown("라벨 결과를 확인하고 학습용 train/val 데이터셋을 구성합니다.")

    prompts = gr.Textbox(label="클래스 이름", placeholder="person, car")
    if label_prompts is not None:
        label_prompts.change(lambda x: gr.update(value=x), inputs=label_prompts, outputs=prompts)

    with gr.Row():
        filter_empty = gr.Checkbox(label="라벨 없는 프레임 제외", value=True)
        val_ratio = gr.Slider(0.05, 0.5, value=0.2, step=0.05, label="검증 비율")

    with gr.Row():
        load_btn = gr.Button("미리보기 불러오기", variant="secondary")
        class_btn = gr.Button("클래스 확인")
        build_btn = gr.Button("데이터셋 구성", variant="primary")

    class_info = gr.Markdown("클래스 확인을 누르세요.")
    gallery = gr.Gallery(label="프레임별 라벨", columns=4, height=420)
    stats = gr.Textbox(label="통계", interactive=False)

    selected_stem = gr.State("")
    detail = gr.Image(label="선택 이미지", type="numpy")
    selected_msg = gr.Textbox(label="선택 상태", interactive=False)
    delete_btn = gr.Button("선택한 이미지 삭제", variant="stop")
    build_status = gr.Textbox(label="구성 상태", lines=8, interactive=False)

    load_btn.click(dataset.load_preview, inputs=[prompts, filter_empty], outputs=[gallery, stats])
    class_btn.click(_class_summary, inputs=prompts, outputs=class_info)
    gallery.select(_select_frame, inputs=[prompts, filter_empty], outputs=[detail, selected_stem, selected_msg])
    delete_btn.click(
        dataset.delete_frame,
        inputs=[selected_stem, prompts, filter_empty],
        outputs=[gallery, stats, detail, selected_stem, selected_msg],
    )
    build_btn.click(dataset.build_dataset, inputs=[prompts, val_ratio, filter_empty], outputs=build_status)
