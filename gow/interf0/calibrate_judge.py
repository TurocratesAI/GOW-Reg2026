#!/usr/bin/env python3
"""
interf0 (Metric-B) calibration against the REAL Qwen3-14B judge. Metric-B is text-only and reduces to gate
accuracy: B1 = background answer refuses (CORRECT), B2 = two identical tissue answers are SAME, B3 = tissue vs
background answers DIFFER and the background is a correct refusal. This confirms our two fixed answer strings
score B1=B2=B3=1.0 under the actual judge (the one open item in task #11). Judge loaded bf16 with device_map=auto
so it fits across the two GPUs on-GPU (no CPU offload). No training; inference-only.

  python gow/interf0/calibrate_judge.py
"""
import glob, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scorer"))
sys.path.insert(0, HERE)


def main():
    snaps = glob.glob("/home/swapnil/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/*/")
    assert snaps, "Qwen3-14B not found in HF cache"
    model_path = snaps[0]
    from evaluate_metrics import LocalQwenJudgeLLM, VisualGroundingEvaluator, score_or_zero, score_or_zero_b3
    from respond import BG_ANSWER, TISSUE_ANSWER
    Q = "Is tissue visible in this ROI? Answer briefly"
    print(f"[calib] loading Qwen3-14B judge from {model_path} (device_map=auto) ...", flush=True)
    judge = LocalQwenJudgeLLM(model_path=model_path, device="cuda", max_new_tokens=16384)
    ev = VisualGroundingEvaluator(judge_model=judge)

    print("[calib] B1 background-refusal ...", flush=True)
    b1r = ev.judge_background(Q, BG_ANSWER); b1 = score_or_zero(b1r)
    print("[calib] B2 stability (tissue vs itself) ...", flush=True)
    b2r = ev.judge_similarity(Q, TISSUE_ANSWER, TISSUE_ANSWER); b2 = score_or_zero(b2r)
    print("[calib] B3 discrimination (tissue vs background) ...", flush=True)
    sim = ev.judge_similarity(Q, TISSUE_ANSWER, BG_ANSWER)
    bg = ev.judge_background(Q, BG_ANSWER)
    b3 = score_or_zero_b3(sim) * score_or_zero(bg)

    print(f"\n  B1 background-refusal : raw={b1r}  score={b1}   (want 1.0 = judge says CORRECT)")
    print(f"  B2 stability          : raw={b2r}  score={b2}   (want 1.0 = judge says SAME)")
    print(f"  B3 discrimination     : sim={sim}  bg={bg}  score={b3}   (want 1.0 = DIFFERENT and bg CORRECT)")
    ok = (b1 == 1.0 and b2 == 1.0 and b3 == 1.0)
    print(f"\n[calib] {'PASS - interf0 wording scores Metric-B = 1.0 on the real judge' if ok else 'CHECK - a verdict missed (tag/wording); inspect above'}")


if __name__ == "__main__":
    main()
