# EconMedia model training

The final task predicts one GPT-labeled primary economic topic and one overall tone from only the headline and abstract. The `multiple` primary-topic class is retained because it is an explicit value in the annotations. Rows skipped by GPT relevance screening are masked rather than converted to negative or neutral labels.

## Reproduce

Activate the project environment, then run from the repository root:

```powershell
python -m pip install -r requirements-training-lock.txt
python scripts/prepare_finetuning_data.py
python scripts/analyze_token_lengths.py
python scripts/train_finbert.py
python scripts/train_t5.py --per-device-batch-size 4
python scripts/build_finetuning_report.py
python -m unittest tests.test_finetuning
```

The effective batch sizes remain 32 for FinBERT and 16 for T5. On an 8 GB GPU the recorded per-device batches are 4 for both models, using gradient accumulation of 8 and 4 respectively. T5 uses BF16 because its FP16 smoke test produced non-finite loss; the training code aborts on any non-finite T5 loss.

## Inference example

```powershell
python scripts/predict.py --model both --headline "Hiring accelerated as inflation eased" --abstract "U.S. employers added jobs while consumer price growth slowed in September."
```

Final models are stored in `models/econmediabert/` and `models/econmedia-t5/`. Detailed validation histories, fixed-label per-class metrics, confusion matrices, predictions, parsing failures, and error cases are under `outputs/finetuning/`.

All reported scores are agreement with GPT-generated labels, not accuracy against an independent human gold standard.
