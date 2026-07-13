"""
REG2026 Challenge - Algorithm Entry Point
=========================================

This file is the container's entrypoint (see Dockerfile). It detects which interface is active and
calls the right handler. It follows the ORGANIZERS' template shape deliberately: load -> predict ->
write -> exit, in ONE process. No subprocess supervisor, no watchdog/memcap threads, no os._exit ---
those were unique to us and added failure surface a vanilla container does not have. The only safety
kept is at the value level: predict_chain_of_thought already routes every failure to a valid
normalized chain, and each handler wraps inference in a try/except -> SAFE fallback so a single bad
case can never crash the run. Huge slides are bounded cooperatively inside extract_slide (a monotonic
deadline checked at Python loop boundaries), so inference finishes and writes well before the kill.

  Interface 0 - Visual Grounding (Metric B)
    Output : visual-context-response.json  - a plain JSON string
  Interface 1 - Workflow Reasoning (Metric A)
    Output : chain-of-thought.json  - a JSON array of {question, answer, next_question}
"""

from core import (
    INPUT_PATH,
    OUTPUT_PATH,
    get_interface_key,
    write_json_file,
    show_torch_cuda_info,
)
from src.interf0.model import predict_visual_context_response, SAFE_RESPONSE
from src.interf1.model import predict_chain_of_thought, SAFE_CHAIN


def run():
    interface_key = get_interface_key()
    handler = {
        (
            "histopathology-region-of-interest-thumbnail",
            "visual-context-question",
        ): interf0_handler,
        ("whole-slide-image",): interf1_handler,
    }[interface_key]
    return handler()


# ---------------------------------------------------------------------------
# Interface 0 - Visual Grounding
# ---------------------------------------------------------------------------

def interf0_handler():
    import traceback
    question_path  = INPUT_PATH / "visual-context-question.json"
    roi_image_path = INPUT_PATH / "histopathology-region-of-interest-thumbnail.jpeg"
    output_path    = OUTPUT_PATH / "visual-context-response.json"
    print(f"[interf0] Question path : {question_path}")
    print(f"[interf0] ROI path      : {roi_image_path}")
    try:
        answer = predict_visual_context_response(question_path=question_path, roi_image_path=roi_image_path)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("empty answer")
    except Exception as e:                                    # never let one ROI crash the whole submission
        print(f"[interf0] HANDLER ERROR: {e!r} -> SAFE_RESPONSE", flush=True); traceback.print_exc()
        answer = SAFE_RESPONSE
    write_json_file(location=output_path, content=answer)
    print(f"[interf0] Answer written: {answer}")
    return 0


# ---------------------------------------------------------------------------
# Interface 1 - Workflow Reasoning
# ---------------------------------------------------------------------------

def interf1_handler():
    import traceback
    output_path = OUTPUT_PATH / "chain-of-thought.json"
    show_torch_cuda_info()

    # Locate the WSI robustly (platform serves /input/images/whole-slide-image/<uid>.tiff, but accept any
    # WSI extension / nesting so a naming quirk never crashes the case).
    img_root = INPUT_PATH / "images"
    wsi_files = []
    for ext in ("*.tiff", "*.tif", "*.svs", "*.ndpi", "*.mrxs", "*.scn", "*.vms", "*.bif", "*.dcm", "*.qptiff"):
        wsi_files += sorted((img_root / "whole-slide-image").glob(ext))
    if not wsi_files:
        wsi_files += list(img_root.rglob("*.tif*")) or [p for p in img_root.rglob("*") if p.is_file()]
    if not wsi_files:
        wsi_files = [p for p in INPUT_PATH.rglob("*") if p.is_file() and p.suffix.lower() in
                     (".tiff", ".tif", ".svs", ".ndpi", ".mrxs")]

    try:
        if not wsi_files:
            raise FileNotFoundError(f"no WSI file located under {img_root}")
        chain_of_thought = predict_chain_of_thought(wsi_path=wsi_files[0])   # already normalized; never raises
        if not isinstance(chain_of_thought, list) or not chain_of_thought:
            raise ValueError("empty chain")
    except Exception as e:                                    # never let one slide crash the whole submission
        print(f"[interf1] HANDLER ERROR: {e!r} -> SAFE_CHAIN", flush=True); traceback.print_exc()
        chain_of_thought = SAFE_CHAIN

    # Print the exact chain we are about to write + an explicit schema self-check, so a failing case's
    # log names the precise violation (organizer request: verify what is produced before it is saved).
    import json as _json
    _slide = wsi_files[0].name if wsi_files else "?"
    print(f"[interf1] chain-of-thought for {_slide} ({len(chain_of_thought)} steps): "
          f"{_json.dumps(chain_of_thought, ensure_ascii=False)}", flush=True)
    _anom = []
    if not isinstance(chain_of_thought, list) or not chain_of_thought:
        _anom.append("not a non-empty array")
    for _i, _st in enumerate(chain_of_thought if isinstance(chain_of_thought, list) else []):
        if not isinstance(_st, dict):
            _anom.append(f"step{_i}:not-object"); continue
        for _k in ("question", "answer", "next_question"):
            if _k not in _st:
                _anom.append(f"step{_i}:missing-{_k}")
            elif not isinstance(_st[_k], str):
                _anom.append(f"step{_i}:{_k}-not-string")
        if isinstance(_st.get("question"), str) and not _st["question"].strip():
            _anom.append(f"step{_i}:empty-question")
        if isinstance(_st.get("answer"), str) and not _st["answer"].strip():
            _anom.append(f"step{_i}:empty-answer")
    print(f"[interf1] schema self-check: {'OK' if not _anom else 'ANOMALIES -> ' + '; '.join(_anom)}", flush=True)

    write_json_file(location=output_path, content=chain_of_thought)
    print(f"[interf1] Chain-of-thought written ({len(chain_of_thought)} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
