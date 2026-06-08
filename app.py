import gradio as gr
from pipeline import extractor, labeler, dataset, trainer, inference


def run_capture(source_type, youtube_url, capture_fps):
    yield from extractor.capture(source_type, youtube_url, int(capture_fps))


def run_label(prompts_str, conf):
    yield from labeler.label(prompts_str, float(conf))


with gr.Blocks(title="YOLO 파이프라인", theme=gr.themes.Default()) as demo:

    # ── Tab 1: 프레임 추출 ──────────────────────────────────────
    with gr.Tab("1. 프레임 추출"):
        gr.Markdown("### 소스 선택 & 프레임 500장 추출")

        with gr.Row():
            source_type = gr.Radio(
                choices=["YouTube URL", "웹캠"],
                value="YouTube URL",
                label="소스",
            )
            capture_fps = gr.Slider(
                minimum=1, maximum=30, value=5, step=1,
                label="캡처 FPS (초당 저장 프레임 수)",
            )

        youtube_url = gr.Textbox(
            placeholder="https://www.youtube.com/watch?v=...",
            label="YouTube URL",
            visible=True,
        )

        source_type.change(
            fn=lambda s: gr.update(visible=(s == "YouTube URL")),
            inputs=source_type,
            outputs=youtube_url,
        )

        with gr.Row():
            cap_start_btn = gr.Button("캡처 시작", variant="primary")
            cap_stop_btn  = gr.Button("중지", variant="stop")

        cap_preview = gr.Image(label="실시간 미리보기", type="numpy")
        cap_status  = gr.Textbox(label="상태", interactive=False)

        capture_event = cap_start_btn.click(
            fn=run_capture,
            inputs=[source_type, youtube_url, capture_fps],
            outputs=[cap_preview, cap_status],
        )
        cap_stop_btn.click(
            fn=extractor.stop,
            cancels=[capture_event],
        )

    # ── Tab 2: SAM3 오토라벨링 ──────────────────────────────────
    with gr.Tab("2. SAM3 오토라벨링"):
        gr.Markdown("### SAM3 텍스트 프롬프트로 자동 라벨링")

        prompts_input = gr.Textbox(
            placeholder="person, car, bicycle",
            label="클래스 프롬프트 (쉼표 구분)",
            value="",
        )
        conf_slider = gr.Slider(
            minimum=0.05, maximum=0.9, value=0.25, step=0.05,
            label="신뢰도 임계값 (conf)",
        )

        with gr.Row():
            label_start_btn = gr.Button("오토라벨링 시작", variant="primary")
            label_stop_btn  = gr.Button("중지", variant="stop")

        label_preview = gr.Image(label="라벨링 미리보기", type="numpy")
        label_status  = gr.Textbox(label="상태", interactive=False)

        label_event = label_start_btn.click(
            fn=run_label,
            inputs=[prompts_input, conf_slider],
            outputs=[label_preview, label_status],
        )
        label_stop_btn.click(
            fn=labeler.stop,
            cancels=[label_event],
        )


    # ── Tab 3: 데이터셋 검토 & 구성 ────────────────────────────────
    with gr.Tab("3. 데이터셋 구성"):
        gr.Markdown("### 라벨링 결과 검토 후 train/val 분할")

        ds_prompts = gr.Textbox(
            placeholder="person, car, bicycle",
            label="클래스 프롬프트 (Tab 2와 동일하게 입력)",
        )

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
            label="프레임별 라벨 확인",
            columns=4,
            height=500,
            object_fit="contain",
        )

        preview_btn.click(
            fn=dataset.load_preview,
            inputs=[ds_prompts, filter_empty_chk],
            outputs=[ds_gallery, ds_stats],
        )

        gr.Markdown("---")
        build_btn    = gr.Button("데이터셋 구성 (train/val 분할)", variant="primary")
        build_status = gr.Textbox(label="진행 상태", interactive=False)

        build_btn.click(
            fn=dataset.build_dataset,
            inputs=[ds_prompts, val_ratio_slider, filter_empty_chk],
            outputs=build_status,
        )

    # ── Tab 4: YOLO 학습 ────────────────────────────────────────
    with gr.Tab("4. YOLO 학습"):
        gr.Markdown("### YOLO 모델 학습")
        gr.Markdown(
            "> **파라미터 안내** — 아래 값은 ultralytics 기본값 기준이며 최적화된 값이 아닙니다. "
            "데이터셋 크기·GPU 환경에 따라 조정하세요."
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
            choices=["auto", "cpu", "0"],
            value="auto",
            label="Device",
            info="auto: GPU 있으면 자동 사용 / cpu: CPU 강제 (느림) / 0: 첫 번째 GPU 지정",
        )

        with gr.Row():
            train_start_btn = gr.Button("학습 시작", variant="primary")
            train_stop_btn  = gr.Button("중지", variant="stop")

        train_log = gr.Textbox(
            label="학습 로그",
            interactive=False,
            lines=20,
            max_lines=20,
            autoscroll=True,
        )

        train_event = train_start_btn.click(
            fn=trainer.train,
            inputs=[epochs_slider, imgsz_slider, batch_slider, lr0_slider, device_radio],
            outputs=train_log,
        )
        train_stop_btn.click(
            fn=trainer.stop,
            cancels=[train_event],
        )

    # ── Tab 5: 추론 ─────────────────────────────────────────────
    with gr.Tab("5. 추론"):
        gr.Markdown("### 학습된 모델로 실시간 추론")

        inf_model_path = gr.Textbox(
            placeholder="비워두면 runs/detect/ 에서 최신 best.pt 자동 탐색",
            label="모델 경로 (best.pt)",
        )
        with gr.Row():
            inf_conf = gr.Slider(
                minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                label="신뢰도 임계값 (conf)",
            )
            inf_skip = gr.Slider(
                minimum=1, maximum=10, value=3, step=1,
                label="추론 간격 (N프레임마다 1회)",
                info="1=매 프레임 추론(느림), 3=3프레임마다 추론(권장), 10=빠르지만 부정확",
            )

        with gr.Row():
            inf_source_type = gr.Radio(
                choices=["YouTube URL", "웹캠"],
                value="YouTube URL",
                label="소스",
            )

        inf_youtube_url = gr.Textbox(
            placeholder="https://www.youtube.com/watch?v=...",
            label="YouTube URL",
            visible=True,
        )

        inf_source_type.change(
            fn=lambda s: gr.update(visible=(s == "YouTube URL")),
            inputs=inf_source_type,
            outputs=inf_youtube_url,
        )

        with gr.Row():
            inf_start_btn = gr.Button("추론 시작", variant="primary")
            inf_stop_btn  = gr.Button("중지", variant="stop")

        inf_preview = gr.Image(label="추론 결과", type="numpy", streaming=True)
        inf_status  = gr.Textbox(label="상태", interactive=False)

        inf_event = inf_start_btn.click(
            fn=inference.predict,
            inputs=[inf_model_path, inf_source_type, inf_youtube_url, inf_conf, inf_skip],
            outputs=[inf_preview, inf_status],
        )
        inf_stop_btn.click(
            fn=inference.stop,
            cancels=[inf_event],
        )


if __name__ == "__main__":
    demo.launch()
