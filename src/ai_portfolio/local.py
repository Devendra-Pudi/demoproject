"""Offline text assistant CLI. Pull weights beforehand; inference uses only Ollama."""
import argparse
import json

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    try:
        with httpx.stream("POST", args.url + "/api/generate", json={
            "model": args.model, "prompt": args.prompt, "stream": True,
            "options": {"temperature": 0, "num_predict": 512},
        }, timeout=120) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    item = json.loads(line)
                    if "error" in item:
                        raise RuntimeError(item["error"])
                    print(item.get("response", ""), end="", flush=True)
        print()
    except (httpx.HTTPError, RuntimeError) as exc:
        parser.exit(1, f"Ollama failed: {exc}\n")


if __name__ == "__main__":
    main()
