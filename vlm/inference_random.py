import torch
import json
import os
import argparse
from pathlib import Path
from PIL import Image
import random

# store paths
os.environ["HF_HOME"]       = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface"
os.environ["HF_HUB_CACHE"]  = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface/hub"
os.environ["XDG_CACHE_HOME"]= "/ubc/cs/research/nlp-raid/students/kwang67/.cache"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import load_model

# Paths 
TRAIN_JSON      = "../data/ChartQA_data/train/train_human_preprocessed.json" # Note: we use the same test json for training samples in few-shot setting, but we will retrieve different questions for the same chart as examples.
TEST_JSON       = "../data/ChartQA_data/test/test_human_preprocessed.json"
TRAIN_IMG_DIR   = "../data/ChartQA_data/train/png"
TEST_IMG_DIR    = "../data/ChartQA_data/test/png"
TRAIN_HEATMAP   = "../data/saliency_maps/ChartQA_train"
TEST_HEATMAP    = "../data/saliency_maps/ChartQA_test" # the saliency map dir for inference, you can change to the one you want.
MAX_SAMPLES     = 100



# Prompt builders 
def build_prompt_zeroshot(question: str, chart_img, heatmap_img=None) -> list:
    if heatmap_img is not None:
        system = {
            "role": "system",
            "content": [
                {"type": "text", "text": 
                "You are an expert chart question answering assistant.\n"
                "You will be given a chart image and a saliency map overlaid on the chart.\n"
                "The saliency map represents human attention when answering the question, highlighting regions humans are likely to focus on.\n"
                "Prioritize information in highlighted regions when forming your answer.\n"
                "Use only information in the chart. Do not use external knowledge or assumptions beyond the chart.\n"
                "Return ONLY the final answer. Do not include explanation or reasoning.\n"
                }
            ]
        }

        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": chart_img},
                {"type": "image", "image": heatmap_img},
                {"type": "text", "text": 
                "The first image is a chart.\n"
                "The second image is a saliency map overlaid on the chart, indicating regions likely relevant to the question.\n\n"
                f"Question: {question}\n\n"
                }
            ]
        }
    else:
        system = {
            "role": "system",
            "content": [
                {"type": "text", "text": 
                "You are an expert chart question answering assistant.\n"
                "You will be given a chart image.\n"
                "Your task is to answer questions using only information in the chart.\n"
                "Do not use external knowledge or assumptions beyond the chart.\n"
                "Return ONLY the final answer. Do not include explanation or reasoning.\n"
                }
            ]
        }
            
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": chart_img},
                {"type": "text", "text": 
                "Answer the question based only on the chart image.\n"
                f"Question: {question}\n\n"
                }
            ]
        }

    return [system, user]

def build_prompt_fewshot(question: str, examples: list, chart_img, heatmap_img=None) -> list:

    if heatmap_img is not None:
        prompt = [{
            "role": "system",
            "content": [
                {"type": "text", "text":
                "You are an expert chart question answering assistant.\n"
                "You will be given several examples, each containing a chart, a saliency map, a question, and a final answer.\n"
                "The saliency map represents human attention when answering the question, highlighting regions humans are likely to focus on.\n"
                "Use the saliency map as a visual attention guide for locating relevant regions in the chart.\n"
                "Learn the mapping pattern from these examples and apply it to the final question.\n"
                "Output ONLY the final answer. Do not include explanations or any extra text.\n"
                }
            ]
        }]

    else:
        prompt = [{
            "role": "system",
            "content": [
                {"type": "text", "text":
                "You are an expert chart question answering assistant.\n"
                "You will be given several examples, each containing a chart, a question, and a correct final answer.\n"
                "Your task is to learn the pattern from the examples and answer the final question.\n"
                "Only output the final answer. Do not include any explanation or extra text.\n"
                }
            ]
        }]

    user_content = []
    for i, ex in enumerate(examples, start = 1):
        if heatmap_img is not None:
            user_content.append({"type": "image", "image": ex["chart_img"]})
            user_content.append({"type": "image", "image": ex["heatmap_img"]})
            user_content.append({"type": "text", "text":
                f"Example {i}:\n"
                "Given the chart and its saliency map, answer the following question.\n"
                f"Question: {ex['query']}\n"
                f"Answer: {ex['label']}\n"
            })
                
        else:
            user_content.append({"type": "image", "image": ex["chart_img"]})
            user_content.append({"type": "text", "text":
                f"Example {i}:\n"
                "Given the chart, answer the question.\n"
                f"Question: {ex['query']}\n"
                f"Answer: {ex['label']}\n"
            })
  
    if heatmap_img is not None:
        user_content.append({"type": "image", "image": chart_img})
        user_content.append({"type": "image", "image": heatmap_img})
        user_content.append({"type": "text", "text":
            "Given the chart and its saliency map, answer the following question.\n"
            f"Question: {question}\n"
        })

    else:
        user_content.append({"type": "image", "image": chart_img})
        user_content.append({"type": "text", "text":
            "Given the chart, answer the question.\n"
            f"Question: {question}\n"
        })
    
    prompt.append({"role": "user", "content": user_content})

    return prompt


# Few-shot example retrieval, here we simply retrieve other questions for the same chart.
def retrieve_examples(train_samples, current_sample) -> list:
    is_numerical = current_sample["is_numerical"]
    is_year      = current_sample["is_year"]

    matching = [
        s for s in train_samples
        if s["is_numerical"] == is_numerical and s["is_year"] == is_year
    ]

    samples = random.sample(matching, 3)
    print(f"Samples: {samples}")
    return samples # json item for the train examples with keys: "imgname", "query", "label", "is_numerical", "saliency_map" 

def load_example_images(examples, img_dir, heatmap_dir):
    loaded  = []
    for ex in examples:
        chart_img   = Image.open(os.path.join(img_dir, ex["imgname"])).convert("RGB")
        heatmap_img = Image.open(os.path.join(heatmap_dir, ex["saliency_map"])).convert("RGB")
        loaded.append({
            **ex,
            "chart_img":   chart_img,
            "heatmap_img": heatmap_img,
        })
    return loaded #json item with keys: "imgname", "query", "label", "is_numerical", "saliency_map", "chart_img", "heatmap_img" (real images)



def run_inference(model, samples, train_samples, setting, use_saliency):
    results       = []

    for i, sample in enumerate(samples):
        # load from test json file
        imgname   = sample["imgname"]
        question  = sample["query"]
        gt_answer = sample["label"]
        is_numerical = sample["is_numerical"]
        is_year = sample["is_year"]
        saliency_map = sample["saliency_map"] if use_saliency else None

        chart_img   = Image.open(os.path.join(TEST_IMG_DIR, imgname)).convert("RGB")
        heatmap_img = Image.open(os.path.join(TEST_HEATMAP, saliency_map)).convert("RGB") if use_saliency  else None

        if setting == "zeroshot":
            prompt = build_prompt_zeroshot(question, chart_img, heatmap_img if use_saliency else None)
            predicted_answer = model.generate(prompt)

        elif setting == "fewshot":
            raw_examples = retrieve_examples(train_samples, sample)
            examples     = load_example_images(raw_examples, TRAIN_IMG_DIR, TRAIN_HEATMAP)
            prompt = build_prompt_fewshot(question, examples, chart_img, heatmap_img if use_saliency else None)
            print(f"heatmap for this question: {saliency_map}")
            print(f"Prompt for {imgname}:\n{prompt}\n")
            print("-----------------------------------------")
            predicted_answer = model.generate(prompt)

        results.append({
            "imgname":      imgname,
            "saliency_map": saliency_map if use_saliency else None,
            "question":     question,
            "gt_answer":    gt_answer,
            "pred_answer":  predicted_answer,
            "is_numerical": is_numerical,
            "is_year": is_year
            })

        if (i + 1) % 10 == 0:
            print(f"----------{i+1}/{len(samples)} processed.----------")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llava15", "chartr1", "internvl", "qwen3vl", "bespokeminchart"], default="llava15")
    parser.add_argument("--setting", choices=["zeroshot", "fewshot"],  default="zeroshot")
    parser.add_argument("--use_saliency", action="store_true")
    parser.add_argument("--max_samples",  type=int, default=MAX_SAMPLES)
    args = parser.parse_args()

    saliency_tag = "with_saliency" if args.use_saliency else "no_saliency"
    output_path  = f"./updated/{args.model}_{args.setting}_{saliency_tag}.json"

    with open(TEST_JSON, "r") as f:
        samples = json.load(f)[:args.max_samples] # load test samples, samples[0]["imgname"] = "1.png"

    train_samples = []
    if args.setting == "fewshot":
        with open(TRAIN_JSON, "r") as f:
            train_samples = json.load(f)[:100] # load train samples for few-shot retrieval

    model   = load_model(args.model)
    results = run_inference(model, samples, train_samples, args.setting, args.use_saliency)

    Path("./updated").mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

# python inference_random.py --model bespokeminchart  --setting zeroshot --use_saliency
