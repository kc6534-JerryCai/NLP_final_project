from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from econmedia_finetuning.constants import PRIMARY_TOPIC_LABELS, TONE_LABELS
from econmedia_finetuning.data import article_text
from econmedia_finetuning.modeling_econmediabert import EconMediaBertForJointClassification
from econmedia_finetuning.t5_format import build_t5_input, parse_t5_output


def predict_finbert(model_dir: Path, headline: str, abstract: str, device: torch.device) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = EconMediaBertForJointClassification.from_pretrained(model_dir).to(device).eval()
    metadata = json.loads((model_dir / "training_metadata.json").read_text(encoding="utf-8"))
    encoded = tokenizer(article_text(headline, abstract), max_length=metadata["max_length"], truncation=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded)
    primary_prob = torch.softmax(output.primary_topic_logits, dim=-1)[0].float().cpu().tolist()
    tone_prob = torch.softmax(output.tone_logits, dim=-1)[0].float().cpu().tolist()
    return {
        "model": "econmediabert",
        "primary_topic": PRIMARY_TOPIC_LABELS[max(range(len(primary_prob)), key=primary_prob.__getitem__)],
        "overall_tone": TONE_LABELS[max(range(len(tone_prob)), key=tone_prob.__getitem__)],
        "primary_topic_probabilities": dict(zip(PRIMARY_TOPIC_LABELS, primary_prob)),
        "tone_probabilities": dict(zip(TONE_LABELS, tone_prob)),
    }


def predict_t5(model_dir: Path, headline: str, abstract: str, device: torch.device) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device).eval()
    metadata = json.loads((model_dir / "training_metadata.json").read_text(encoding="utf-8"))
    encoded = tokenizer(build_t5_input(headline, abstract), max_length=metadata["max_source_length"], truncation=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=32, do_sample=False, num_beams=1)
    raw = tokenizer.decode(generated[0], skip_special_tokens=True)
    parsed = parse_t5_output(raw)
    return {
        "model": "econmedia-t5",
        "raw_output": raw,
        "format_valid": parsed.valid,
        "primary_topic": parsed.primary_topic,
        "overall_tone": parsed.tone,
        "parse_errors": parsed.errors,
        "unknown_labels": parsed.unknown_labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict primary economic topic and overall tone")
    parser.add_argument("--model", choices=["finbert", "t5", "both"], default="both")
    parser.add_argument("--headline", required=True)
    parser.add_argument("--abstract", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    if args.model in {"finbert", "both"}:
        results.append(predict_finbert(ROOT / "models" / "econmediabert", args.headline, args.abstract, device))
    if args.model in {"t5", "both"}:
        results.append(predict_t5(ROOT / "models" / "econmedia-t5", args.headline, args.abstract, device))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
