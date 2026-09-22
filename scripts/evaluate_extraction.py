"""Measure held-out JSON validity, field accuracy and exact match for base or adapter."""
import argparse
import json
import time
from pathlib import Path


def score(predictions, expected):
    valid = exact = fields = 0
    for text, target in zip(predictions, expected, strict=True):
        try:
            parsed = json.loads(text)
            valid += 1
            exact += parsed == target
            if isinstance(parsed, dict):
                fields += sum(parsed.get(k) == v for k, v in target.items())
        except ValueError:
            pass
    return {"n": len(expected), "json_valid_rate": valid / len(expected),
            "exact_match": exact / len(expected),
            "field_accuracy": fields / sum(len(x) for x in expected)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    cases = [json.loads(line) for line in Path("data/training/heldout.jsonl").read_text().splitlines()]
    predictions, latencies = [], []
    for case in cases:
        prompt = tokenizer.apply_chat_template(case["messages"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.inference_mode():
            result = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        predictions.append(tokenizer.decode(result[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
        latencies.append((time.perf_counter() - start) * 1000)
    report = {"model": args.model, "adapter": args.adapter,
              "metrics": score(predictions, [c["expected"] for c in cases]),
              "predictions": predictions, "latencies_ms": latencies}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
