import torch
import json
import os
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from collections import defaultdict
 
# load environment variables for Hugging Face cache
os.environ["HF_HOME"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface/hub"
os.environ["XDG_CACHE_HOME"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache"

# Paths 
JSON_PATH      = "../ChartQA_data/train_human.json"
IMG_DIR        = "../ChartQA_data/png"
HEATMAP_DIR    = "../eval_results"
OUTPUT_PATH    = "./result/llava15_zeroshot_with_saliency.json"
MAX_SAMPLES    = 50

def load_model():
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/ubc/cs/research/nlp-raid/students/kwang67/.cache"
    )
    processor = AutoProcessor.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        cache_dir="/ubc/cs/research/nlp-raid/students/kwang67/.cache"
    ) # process images and text into a tensor that model can use.
    print("Model loaded: llava-hf/llava-1.5-7b-hf")
    return model, processor
 
 
def build_prompt(question: str) -> str:
    return (
        "USER: <image>\n<image>\n"
        "The first image is a chart that you need to analyze. "
        "The second image is a saliency map highlighting the regions most relevant to the question.\n"
        "Use the saliency map to guide your attention while answering the question given to you.\n\n"
        f"Question: {question}\n"
        "Give a short, direct answer only.\n"
        "ASSISTANT:"
    )
 
 
def run_inference(model, processor, samples, img_dir, heatmap_dir):
    results = []
    query_counter = defaultdict(int)
 
    for i, sample in enumerate(samples):
        imgname = sample["imgname"]
        question = sample["query"]
        gt_answer = sample["label"]
 
        stem = os.path.splitext(imgname)[0]
        q_idx = query_counter[stem]
        query_counter[stem] += 1
 
        # Load images
        chart_path   = os.path.join(img_dir, imgname)
        heatmap_path = os.path.join(heatmap_dir, f"{stem}_Q{q_idx}.png")
 
        chart_img   = Image.open(chart_path).convert("RGB")
        heatmap_img = Image.open(heatmap_path).convert("RGB")
 
        # Build prompt & run model 
        prompt = build_prompt(question)
        inputs = processor(
            text=prompt,
            images=[chart_img, heatmap_img],
            return_tensors="pt"
        ).to("cuda")
 
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
 
        # Decode the newly generated tokens (skip the prompt)
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        predicted_answer = processor.decode(generated, skip_special_tokens=True).strip()
 
        results.append({
            "imgname":   imgname,
            "question":  question,
            "gt_answer": gt_answer,
            "pred_answer": predicted_answer,
            "heatmap":   f"{stem}_Q{q_idx}.png"
        })
 
        # Progress logging every 10 samples
        if (i + 1) % 10 == 0:
            print(f"{i+1} samples have been processed.")
 
    return results
 
 
def main():
    # Load data
    with open(JSON_PATH, "r") as f:
        samples = json.load(f)
    if MAX_SAMPLES:
        samples = samples[:MAX_SAMPLES]
    print(f"Running inference on {len(samples)} samples.")
 
    # Load model
    model, processor = load_model()
 
    # Run
    results = run_inference(model, processor, samples, IMG_DIR, HEATMAP_DIR)
 
    # Save
    Path("./result").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_PATH}")
 
 
if __name__ == "__main__":
    main()
 
