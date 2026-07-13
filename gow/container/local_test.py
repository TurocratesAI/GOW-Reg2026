"""
Local LOGIC test for the two interface functions - no Docker, dev weights.
Proves the wiring (WSI->chain, ROI->answer) end-to-end before the Docker do_test_run.

  CUDA_VISIBLE_DEVICES=0 python gow/container/local_test.py
"""
import os, sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                # core.py, gowcfg.py
os.environ["GOW_SRC"] = str(HERE.parent)                     # reg2026/gow
os.environ.setdefault("MODEL_PATH", "/nonexistent-force-dev-fallbacks")

from src.interf0.model import predict_visual_context_response
from src.interf1.model import predict_chain_of_thought

IN = HERE / "test" / "input"


def test_interf0():
    print("\n" + "=" * 60 + "\nINTERF0 (visual grounding)\n" + "=" * 60)
    ans = predict_visual_context_response(
        question_path=IN / "interf0" / "visual-context-question.json",
        roi_image_path=IN / "interf0" / "histopathology-region-of-interest-thumbnail.jpeg",
    )
    assert isinstance(ans, str) and ans.strip(), "interf0 must return a non-empty string"
    print("  RETURN (str):", repr(ans[:90]))
    print("  OK: non-empty string")


def test_interf1():
    print("\n" + "=" * 60 + "\nINTERF1 (workflow reasoning)\n" + "=" * 60)
    wsi = next((IN / "interf1" / "images" / "whole-slide-image").glob("*.tiff"))
    chain = predict_chain_of_thought(wsi_path=wsi)
    assert isinstance(chain, list) and chain, "interf1 must return a non-empty list"
    for s in chain:
        assert set(s) >= {"question", "answer", "next_question"}, f"bad step keys: {s.keys()}"
    assert chain[-1]["next_question"] == "", "last step next_question must be ''"
    print(f"  RETURN (list): {len(chain)} steps")
    for s in chain[:4]:
        print(f"    Q {s['question'][:44]!r:46} A {s['answer'][:40]!r}")
    print("    ...")
    print(f"    REPORT: {chain[-1]['answer']!r}")
    print("  OK: valid chain, last next_question=''")


if __name__ == "__main__":
    test_interf0()
    test_interf1()
    print("\n[local_test] BOTH INTERFACES OK - logic wired. Next: Docker do_test_run.")
