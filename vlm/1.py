import torch
import json
import os
import argparse
from pathlib import Path
from PIL import Image
from collections import defaultdict

os.environ["HF_HOME"]       = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface"
os.environ["HF_HUB_CACHE"]  = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface/hub"
os.environ["XDG_CACHE_HOME"]= "/ubc/cs/research/nlp-raid/students/kwang67/.cache"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import load_model

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_JSON      = "../ChartQA_data/train/train_human.json"
TEST_JSON       = "../ChartQA_data/test/test_human.json"
TRAIN_IMG_DIR   = "../ChartQA_data/train/png"
TEST_IMG_DIR    = "../ChartQA_data/test/png"
TRAIN_HEATMAP   = "../saliency_maps/ChartQA_train"
TEST_HEATMAP    = "../saliency_maps/ChartQA_test"
MAX_SAMPLES     = 50


# ── Prompt builders ────────────────────────────────────────────────────────────
def build_prompt_zeroshot(question: str, use_saliency: bool) -> str:
    if use_saliency:
        return (
            "USER: <image>\n<image>\n"
            "The first image is a chart. "
            "The second image is a saliency map highlighting regions relevant to the question.\n"
            "Use the saliency map to guide your attention.\n\n"
            f"Question: {question}\n"
            "Give a short, direct answer only.\n"
            "ASSISTANT:"
        )
    else:
        return (
            "USER: <image>\n"
            f"Question: {question}\n"
            "Give a short, direct answer only.\n"
            "ASSISTANT:"
        )


def build_prompt_fewshot(question: str, examples: list, use_saliency: bool) -> tuple[str, list]:
    """Returns (prompt_text, list of example images in order)"""
    prompt = "USER: "
    images = []

    for ex in examples:
        if use_saliency:
            prompt += "<image>\n<image>\n"
            images.append(ex["chart_img"])
            images.append(ex["heatmap_img"])
        else:
            prompt += "<image>\n"
            images.append(ex["chart_img"])
        prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"

    if use_saliency:
        prompt += "<image>\n<image>\n"
    else:
        prompt += "<image>\n"

    prompt += (
        "Use the above examples as reference.\n"
        f"Question: {question}\n"
        "Give a short, direct answer only.\n"
        "ASSISTANT:"
    )
    return prompt, images  # current chart/heatmap appended in run_inference


# ── Few-shot example retrieval ─────────────────────────────────────────────────
def retrieve_examples(train_samples, current_sample) -> list:
    """Same chart (same imgname), different queries."""
    stem = os.path.splitext(current_sample["imgname"])[0]
    return [s for s in train_samples
            if os.path.splitext(s["imgname"])[0] == stem
            and s["query"] != current_sample["query"]]


def load_example_images(examples, img_dir, heatmap_dir):
    """Load PIL images for each example and attach to dict."""
    counter = defaultdict(int)
    loaded  = []
    for ex in examples:
        stem  = os.path.splitext(ex["imgname"])[0]
        q_idx = counter[stem]
        counter[stem] += 1
        chart_img   = Image.open(os.path.join(img_dir, ex["imgname"])).convert("RGB")
        heatmap_img = Image.open(os.path.join(heatmap_dir, f"{stem}_Q{q_idx}.png")).convert("RGB")
        loaded.append({
            **ex,
            "chart_img":   chart_img,
            "heatmap_img": heatmap_img,
            "question":    ex["query"],
            "answer":      ex["label"],
        })
    return loaded


# ── Main inference loop ────────────────────────────────────────────────────────
def run_inference(model, samples, train_samples, mode, use_saliency):
    results       = []
    query_counter = defaultdict(int)

    for i, sample in enumerate(samples):
        imgname   = sample["imgname"]
        question  = sample["query"]
        gt_answer = sample["label"]

        stem  = os.path.splitext(imgname)[0]
        q_idx = query_counter[stem]
        query_counter[stem] += 1

        chart_img   = Image.open(os.path.join(TEST_IMG_DIR, imgname)).convert("RGB")
        heatmap_img = Image.open(os.path.join(TEST_HEATMAP, f"{stem}_Q{q_idx}.png")).convert("RGB") if use_saliency else None

        if mode == "zeroshot":
            prompt = build_prompt_zeroshot(question, use_saliency)
            images = [chart_img, heatmap_img] if use_saliency else [chart_img]

        elif mode == "fewshot":
            raw_examples = retrieve_examples(train_samples, sample)
            examples     = load_example_images(raw_examples, TRAIN_IMG_DIR, TRAIN_HEATMAP)
            prompt, ex_images = build_prompt_fewshot(question, examples, use_saliency)
            images = ex_images + ([chart_img, heatmap_img] if use_saliency else [chart_img])

        predicted_answer = model.generate(prompt, images)

        results.append({
            "imgname":      imgname,
            "question":     question,
            "gt_answer":    gt_answer,
            "pred_answer":  predicted_answer,
            "heatmap":      f"{stem}_Q{q_idx}.png" if use_saliency else None,
        })

        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(samples)} processed.")

    return results


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        choices=["llava15", "internvl"], default="llava15")
    parser.add_argument("--mode",         choices=["zeroshot", "fewshot"],  default="zeroshot")
    parser.add_argument("--use_saliency", action="store_true")
    parser.add_argument("--max_samples",  type=int, default=MAX_SAMPLES)
    args = parser.parse_args()

    saliency_tag = "with_saliency" if args.use_saliency else "no_saliency"
    output_path  = f"./hhhh/{args.model}_{args.mode}_{saliency_tag}.json"

    with open(TEST_JSON, "r") as f:
        samples = json.load(f)[:args.max_samples]

    train_samples = []
    if args.mode == "fewshot":
        with open(TRAIN_JSON, "r") as f:
            train_samples = json.load(f)

    print(f"Model: {args.model} | Mode: {args.mode} | Saliency: {args.use_saliency} | Samples: {len(samples)}")

    model   = load_model(args.model)
    results = run_inference(model, samples, train_samples, args.mode, args.use_saliency)

    Path("./hhhh").mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()