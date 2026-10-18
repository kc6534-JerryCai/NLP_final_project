import json
import unittest
from pathlib import Path

import torch
from transformers import BertConfig

from econmedia_finetuning.modeling_econmediabert import EconMediaBertForJointClassification
from econmedia_finetuning.t5_format import build_t5_target, parse_t5_output


ROOT = Path(__file__).resolve().parents[1]


class FineTuningTests(unittest.TestCase):
    def test_t5_strict_format(self):
        target = build_t5_target("employment", "negative")
        parsed = parse_t5_output(target)
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.primary_topic, "employment")
        self.assertFalse(parse_t5_output("tone: negative; primary_topic: employment").valid)
        self.assertFalse(parse_t5_output("primary_topic: unknown; tone: negative").valid)

    def test_masked_joint_loss_is_finite(self):
        config = BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64)
        config.num_primary_topic_labels = 9
        config.num_tone_labels = 5
        model = EconMediaBertForJointClassification(config)
        output = model(
            input_ids=torch.randint(0, 100, (2, 8)),
            attention_mask=torch.ones((2, 8), dtype=torch.long),
            primary_topic_labels=torch.tensor([1, -1]),
            tone_labels=torch.tensor([2, -1]),
            primary_topic_valid_mask=torch.tensor([1.0, 0.0]),
            tone_valid_mask=torch.tensor([1.0, 0.0]),
        )
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(torch.isfinite(output.primary_topic_loss))
        self.assertTrue(torch.isfinite(output.tone_loss))

    def test_fixed_splits_do_not_overlap(self):
        splits = json.loads((ROOT / "data" / "finetuning" / "article_id_splits.json").read_text(encoding="utf-8"))
        self.assertEqual({name: len(ids) for name, ids in splits.items()}, {"train": 4000, "validation": 500, "test": 500})
        self.assertFalse(set(splits["train"]) & set(splits["validation"]))
        self.assertFalse(set(splits["train"]) & set(splits["test"]))
        self.assertFalse(set(splits["validation"]) & set(splits["test"]))


if __name__ == "__main__":
    unittest.main()
