"""Two-stage contact JSON extraction: LoRA/QLoRA SFT, then DPO on that adapter."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["sft", "dpo"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", help="Required SFT adapter for DPO")
    parser.add_argument("--qlora", action="store_true", help="Requires CUDA and bitsandbytes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3)
    args = parser.parse_args()
    if args.stage == "dpo" and not args.adapter:
        parser.error("DPO must start from --adapter pointing to the SFT output")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    if args.qlora and not torch.cuda.is_available():
        parser.error("QLoRA requires a supported CUDA GPU")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=dtype) if args.qlora else None
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                quantization_config=quantization,
                                                device_map="auto" if args.qlora else None)
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                      task_type="CAUSAL_LM")
    common = dict(output_dir=str(args.output), num_train_epochs=args.epochs,
                  per_device_train_batch_size=1, gradient_accumulation_steps=4,
                  learning_rate=1e-4 if args.stage == "sft" else 5e-6,
                  logging_steps=1, save_strategy="epoch", report_to="none", seed=42,
                  bf16=dtype == torch.bfloat16, gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.stage == "sft":
        data = load_dataset("json", data_files="data/training/sft.jsonl", split="train")
        trainer = SFTTrainer(model=model, processing_class=tokenizer, peft_config=lora,
                             args=SFTConfig(**common, max_seq_length=512), train_dataset=data)
    else:
        model = PeftModel.from_pretrained(model, args.adapter, adapter_name="policy", is_trainable=True)
        # Frozen SFT reference, not the untuned base: DPO compares against the actual SFT policy.
        model.load_adapter(args.adapter, adapter_name="reference", is_trainable=False)
        model.set_adapter("policy")
        data = load_dataset("json", data_files="data/training/dpo.jsonl", split="train")
        trainer = DPOTrainer(model=model, processing_class=tokenizer,
                             args=DPOConfig(**common, beta=0.1, max_length=512, max_prompt_length=256,
                                            model_adapter_name="policy", ref_adapter_name="reference"),
                             train_dataset=data)
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print("Training complete. Evaluate the base, SFT, and DPO checkpoints on the SAME held-out data.")


if __name__ == "__main__":
    main()
