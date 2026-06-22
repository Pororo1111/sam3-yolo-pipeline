import html as _html

import gradio as gr
from pipeline import extractor, labeler, dataset, trainer, inference, zone_monitor, models


def _inherit_folder(source_type, current_files, tab1_files):
    """이미지 폴더 선택 시, 업로드가 비어 있으면 Tab 1의 업로드 파일을 물려받는다."""
    if source_type == "이미지 폴더" and not current_files:
        return gr.update(value=tab1_files or None)
    return gr.update()


def run_capture(source_type, youtube_url, capture_fps, folder_files):
    yield from extractor.capture(source_type, youtube_url, int(capture_fps), folder_files)


def load_classes(label_prompts):
    """라벨을 스캔해 (class_state, 최종 클래스 이름, 로드 기준 Tab2 프롬프트) 반환.

    이름은 Tab 2 프롬프트(label_prompts)를 우선으로 새로 구성한다.
    세 번째 반환값은 "이 프롬프트 기준으로 로드했다"는 추적용 값.
    """
    classes = dataset.scan_classes(label_prompts or "")
    joined = ", ".join(c["name"] for c in classes)
    return classes, joined, (label_prompts or "")


def run_label(prompts_str, conf):
    yield from labeler.label(prompts_str, float(conf))


def on_gallery_select(prompts_str, filter_empty, evt: gr.SelectData):
    """갤러리 선택 이벤트 래퍼 — SelectData 어노테이션으로 evt 주입을 명시."""
    return dataset.select_frame(prompts_str, filter_empty, evt)


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


def run_train(epochs, imgsz, batch, lr0, device):
    for text in trainer.train(epochs, imgsz, batch, lr0, device):
        escaped = _html.escape(text)
        yield f'<div style="{_WRAP_STYLE}"><pre style="{_PRE_STYLE}">{escaped}</pre></div>'


_CSS = (
    ".src-hidden { display: none !important; }"
    ".cls-name-box input { font-weight:600; }"
    # 사이드바 내비게이션 — 세로 풀폭 버튼 형태
    "#pipeline_nav .wrap { flex-direction: column !important; "
    "align-items: stretch !important; gap: 6px !important; }"
    "#pipeline_nav label { width: 100% !important; margin: 0 !important; }"
    # 본문 패널이 좁은 화면에서도 넘치지 않도록
    ".main-panel { max-width: 1100px; margin: 0 auto; width: 100%; }"
)

# 사이드바 내비게이션 단계 (파이프라인 순서) — 표시 이름에서 번호 제거
NAV_STEPS = [
    "프레임 추출",
    "SAM3 오토라벨링",
    "데이터셋 구성",
    "YOLO 학습",
    "추론",
    "침입 감지",
]

with gr.Blocks(title="YOLO 파이프라인") as demo:

    # ── 사이드바 (PC: 고정 / 모바일: 토글 드로어) ────────────────
    with gr.Sidebar(width=260):
        gr.Markdown("## 🦾 YOLO 파이프라인")
        gr.Markdown("컴퓨터 비전 학습·추론 end-to-end 파이프라인")
        nav = gr.Radio(
            choices=NAV_STEPS,
            value=NAV_STEPS[0],
            show_label=False,
            container=False,
            elem_id="pipeline_nav",
        )

    # ── 프레임 추출 ──────────────────────────────────────
    with gr.Column(visible=True, elem_classes=["main-panel"]) as panel_extract:
        gr.Markdown("### 소스 선택 & 프레임 추출 (중지 버튼으로 종료)")

        with gr.Row():
            source_type = gr.Radio(
                choices=["YouTube URL", "웹캠", "이미지 폴더"],
                value="YouTube URL",
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
        )
        with gr.Row(elem_id="tab1_folder_row", elem_classes=["src-hidden"]) as folder_row:
            folder_files = gr.File(
                label="이미지 폴더 업로드 (폴더 또는 여러 파일 선택)",
                file_count="directory",
                type="filepath",
                height=200,
            )

        source_type.change(
            fn=None,
            inputs=source_type,
            outputs=None,
            js="""(s) => {
                const yt = document.getElementById('tab1_youtube_url');
                const fr = document.getElementById('tab1_folder_row');
                if (yt) yt.classList.toggle('src-hidden', s !== 'YouTube URL');
                if (fr) fr.classList.toggle('src-hidden', s !== '이미지 폴더');
            }""",
        )

        with gr.Row():
            cap_start_btn = gr.Button("캡처 시작", variant="primary")
            cap_stop_btn  = gr.Button("중지", variant="stop")

        cap_preview = gr.Image(label="실시간 미리보기", type="numpy")
        cap_status  = gr.Textbox(label="상태", interactive=False)

        capture_event = cap_start_btn.click(
            fn=run_capture,
            inputs=[source_type, youtube_url, capture_fps, folder_files],
            outputs=[cap_preview, cap_status],
        )
        cap_stop_btn.click(
            fn=extractor.stop,
            cancels=[capture_event],
        )

    # ── SAM3 오토라벨링 ──────────────────────────────────
    with gr.Column(visible=False, elem_classes=["main-panel"]) as panel_label:
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
        )

        label_preview = gr.Image(label="전체 라벨링 진행 미리보기", type="numpy")
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
    with gr.Column(visible=False, elem_classes=["main-panel"]) as panel_dataset:
        gr.Markdown("### 라벨링 결과 검토 후 train/val 분할")

        gr.Markdown(
            "#### 클래스 목록 — 각 이름 칸에서 바로 수정하세요\n"
            "라벨에서 감지된 클래스 ID와 객체 수입니다. **이름은 이 칸에서 직접 수정**하면 "
            "되고, 하단의 '최종 클래스 이름'에 자동 반영됩니다. (Tab 2에서 새로 라벨링하면 "
            "새 프롬프트 이름으로 자동 갱신됩니다.)"
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
                    gr.Markdown(f"**ID {c['id']}** · {c['count']}개 객체")
                    nb = gr.Textbox(
                        value=c["name"],
                        show_label=False,
                        container=False,
                        scale=3,
                        elem_classes=["cls-name-box"],
                    )
                name_boxes.append(nb)

            def _sync(*names):
                # ID 순서대로 이름을 모아 최종 클래스 이름 문자열 생성
                return ", ".join((n or "").strip() for n in names)

            # 어느 칸을 수정해도 즉시 최종 클래스 이름에 반영 (state 미변경 → 리렌더/포커스 유실 없음)
            for nb in name_boxes:
                nb.change(_sync, inputs=name_boxes, outputs=ds_prompts)

        ds_prompts = gr.Textbox(
            label="최종 클래스 이름 (ID 0,1,2… 순서 · 위 이름 칸에서 자동 생성)",
            placeholder="위에서 클래스를 불러오면 자동으로 채워집니다.",
            interactive=False,
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

    # ── YOLO 학습 ────────────────────────────────────────
    with gr.Column(visible=False, elem_classes=["main-panel"]) as panel_train:
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

        train_log = gr.HTML(label="학습 로그")

        train_event = train_start_btn.click(
            fn=run_train,
            inputs=[epochs_slider, imgsz_slider, batch_slider, lr0_slider, device_radio],
            outputs=train_log,
        )
        train_stop_btn.click(
            fn=trainer.stop,
            cancels=[train_event],
        )

    # ── 추론 ─────────────────────────────────────────────
    with gr.Column(visible=False, elem_classes=["main-panel"]) as panel_infer:
        gr.Markdown("### 학습된 모델로 실시간 추론")

        inf_model_path = gr.Textbox(
            placeholder="비워두면 runs/detect/ 에서 최신 best.pt 자동 탐색",
            label="모델 경로 (best.pt)",
        )
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
                choices=["YouTube URL", "웹캠", "이미지 폴더"],
                value="YouTube URL",
                label="소스",
            )

        inf_youtube_url = gr.Textbox(
            placeholder="https://www.youtube.com/watch?v=...",
            label="YouTube URL",
            elem_id="tab5_youtube_url",
        )
        with gr.Row(elem_id="tab5_folder_row", elem_classes=["src-hidden"]) as inf_folder_row:
            inf_folder_files = gr.File(
                label="이미지 폴더 업로드 (비우면 Tab 1 업로드 자동 사용)",
                file_count="directory",
                type="filepath",
                height=200,
            )

        inf_source_type.change(
            fn=None,
            inputs=inf_source_type,
            outputs=None,
            js="""(s) => {
                const yt = document.getElementById('tab5_youtube_url');
                const fr = document.getElementById('tab5_folder_row');
                if (yt) yt.classList.toggle('src-hidden', s !== 'YouTube URL');
                if (fr) fr.classList.toggle('src-hidden', s !== '이미지 폴더');
            }""",
        )
        inf_source_type.change(
            fn=_inherit_folder,
            inputs=[inf_source_type, inf_folder_files, folder_files],
            outputs=inf_folder_files,
            show_progress="hidden",
        )

        with gr.Row():
            inf_start_btn = gr.Button("추론 시작", variant="primary")
            inf_stop_btn  = gr.Button("중지", variant="stop")

        inf_preview = gr.Image(label="추론 결과", type="numpy", streaming=True)
        inf_status  = gr.Textbox(label="상태", interactive=False)

        inf_event = inf_start_btn.click(
            fn=inference.predict,
            inputs=[inf_model_path, inf_source_type, inf_youtube_url, inf_conf, inf_skip, inf_folder_files],
            outputs=[inf_preview, inf_status],
        )
        inf_stop_btn.click(
            fn=inference.stop,
            cancels=[inf_event],
        )


    # ── 침입 감지 ────────────────────────────────────────
    with gr.Column(visible=False, elem_classes=["main-panel"]) as panel_zone:
        gr.Markdown("### 로컬 LLM 영역 설정 & 실시간 침입 감지")

        zm_model_path = gr.Textbox(
            placeholder="비워두면 runs/detect/ 에서 최신 best.pt 자동 탐색",
            label="YOLO 모델 경로 (best.pt)",
        )

        with gr.Row():
            zm_conf = gr.Slider(
                minimum=0.05, maximum=0.95, value=0.15, step=0.05,
                label="신뢰도 임계값 (conf)",
            )
            zm_skip = gr.Slider(
                minimum=1, maximum=10, value=3, step=1,
                label="추론 간격 (N프레임마다 1회)",
            )

        with gr.Row():
            zm_source_type = gr.Radio(
                choices=["YouTube URL", "웹캠", "이미지 폴더"],
                value="YouTube URL",
                label="소스",
            )

        zm_youtube_url = gr.Textbox(
            placeholder="https://www.youtube.com/watch?v=...",
            label="YouTube URL",
            elem_id="tab6_youtube_url",
        )
        with gr.Row(elem_id="tab6_folder_row", elem_classes=["src-hidden"]) as zm_folder_row:
            zm_folder_files = gr.File(
                label="이미지 폴더 업로드 (비우면 Tab 1 업로드 자동 사용)",
                file_count="directory",
                type="filepath",
                height=200,
            )

        zm_source_type.change(
            fn=None,
            inputs=zm_source_type,
            outputs=None,
            js="""(s) => {
                const yt = document.getElementById('tab6_youtube_url');
                const fr = document.getElementById('tab6_folder_row');
                if (yt) yt.classList.toggle('src-hidden', s !== 'YouTube URL');
                if (fr) fr.classList.toggle('src-hidden', s !== '이미지 폴더');
            }""",
        )
        zm_source_type.change(
            fn=_inherit_folder,
            inputs=[zm_source_type, zm_folder_files, folder_files],
            outputs=zm_folder_files,
            show_progress="hidden",
        )

        with gr.Row():
            zm_start_btn = gr.Button("스트림 시작", variant="primary")
            zm_stop_btn  = gr.Button("중지 / 초기화", variant="stop")

        zm_preview = gr.Image(label="실시간 영상", type="numpy", streaming=True)
        zm_stream_status = gr.Textbox(label="상태", interactive=False)

        gr.Markdown("---")
        gr.Markdown("#### 영역 설정 (로컬 LLM)")

        with gr.Row():
            zm_prompt = gr.Textbox(
                placeholder="예: 문 앞쪽 출입 구역, 컨베이어 벨트 위",
                label="감시 영역 설명",
                scale=3,
            )
            zm_model = gr.Textbox(
                value="gemma4:e4b",
                placeholder="gemma4:e4b",
                label="Ollama 모델",
                scale=1,
            )

        zm_set_btn = gr.Button("영역 설정 (LLM 분석)", variant="secondary")

        zm_zone_status  = gr.Textbox(label="영역 설정 결과", interactive=False)
        zm_llm_log      = gr.Code(label="LLM 응답 (JSON)", language="json", interactive=False)

        zm_stream_event = zm_start_btn.click(
            fn=zone_monitor.stream,
            inputs=[zm_source_type, zm_youtube_url, zm_model_path, zm_conf, zm_skip, zm_folder_files],
            outputs=[zm_preview, zm_stream_status],
        )
        zm_stop_btn.click(
            fn=zone_monitor.reset,
            cancels=[zm_stream_event],
        )
        zm_set_btn.click(
            fn=zone_monitor.set_zone,
            inputs=[zm_prompt, zm_model],
            outputs=[zm_zone_status, zm_llm_log],
        )

    # ── 사이드바 내비게이션 → 패널 전환 ──────────────────────────
    _PANELS = [panel_extract, panel_label, panel_dataset,
               panel_train, panel_infer, panel_zone]

    def _switch_panel(choice):
        return [gr.update(visible=(step == choice)) for step in NAV_STEPS]

    nav.change(_switch_panel, inputs=nav, outputs=_PANELS)

    # 패널 전환 직후 강제 리페인트 — Gradio 6에서 visible 토글로 처음 노출된
    # Column이 레이아웃되지 않아 검정/빈 화면으로 보이는 문제 회피
    # (resize 이벤트를 dispatch해 Svelte/브라우저의 리플로우를 강제)
    nav.change(
        fn=None,
        inputs=None,
        outputs=None,
        js="""() => {
            requestAnimationFrame(() => {
                window.dispatchEvent(new Event('resize'));
                setTimeout(() => window.dispatchEvent(new Event('resize')), 60);
            });
        }""",
    )

    def _maybe_load_classes(choice, label_prompts, last_label, class_state):
        """데이터셋 단계 진입 시 클래스 목록 로드.

        - Tab 2 프롬프트가 직전 로드와 다르거나(재라벨링) 아직 로드된 적 없으면 새로 로드
        - 그 외(단순 탭 전환)에는 그대로 두어 사용자가 칸에서 편집한 이름을 보존
        """
        if choice != "데이터셋 구성":
            return gr.update(), gr.update(), gr.update()
        if class_state and (label_prompts or "") == (last_label or ""):
            return gr.update(), gr.update(), gr.update()
        return load_classes(label_prompts)

    nav.change(
        _maybe_load_classes,
        inputs=[nav, prompts_input, ds_last_label, ds_class_state],
        outputs=[ds_class_state, ds_prompts, ds_last_label],
    )


if __name__ == "__main__":
    # 시작 시 학습용 베이스 모델(yolo26n.pt)이 models/ 에 없으면 자동 다운로드
    models.ensure_models()
    demo.queue().launch(theme=gr.themes.Default(), css=_CSS)
