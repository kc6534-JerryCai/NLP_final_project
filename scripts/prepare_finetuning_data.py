from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from econmedia_finetuning.data import prepare_data
from econmedia_finetuning.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe 4000/500/500 fine-tuning splits")
    parser.add_argument("--config", default="config/finetuning.v1.json")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    result = prepare_data(config, ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
