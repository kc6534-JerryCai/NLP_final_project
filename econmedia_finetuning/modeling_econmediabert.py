from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.utils import ModelOutput


@dataclass
class EconMediaBertOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    primary_topic_loss: Optional[torch.Tensor] = None
    tone_loss: Optional[torch.Tensor] = None
    primary_topic_logits: Optional[torch.Tensor] = None
    tone_logits: Optional[torch.Tensor] = None


class EconMediaBertForJointClassification(BertPreTrainedModel):
    """Shared BERT encoder with independent primary-topic and tone heads."""

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_primary_topic_labels = int(getattr(config, "num_primary_topic_labels", 9))
        self.num_tone_labels = int(getattr(config, "num_tone_labels", 5))
        self.primary_topic_loss_weight = float(getattr(config, "primary_topic_loss_weight", 1.0))
        self.tone_loss_weight = float(getattr(config, "tone_loss_weight", 1.0))
        self.bert = BertModel(config)
        dropout_rate = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_rate)
        self.primary_topic_classifier = nn.Linear(config.hidden_size, self.num_primary_topic_labels)
        self.tone_classifier = nn.Linear(config.hidden_size, self.num_tone_labels)
        self.post_init()

    @classmethod
    def from_finbert(cls, model_name: str, local_files_only: bool = False, **config_values):
        encoder = BertModel.from_pretrained(model_name, local_files_only=local_files_only)
        config = encoder.config
        for key, value in config_values.items():
            setattr(config, key, value)
        config.architectures = [cls.__name__]
        model = cls(config)
        model.bert.load_state_dict(encoder.state_dict(), strict=True)
        return model

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
        mask = mask.to(values.dtype)
        if not torch.any(mask > 0):
            return None
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        primary_topic_labels: Optional[torch.Tensor] = None,
        tone_labels: Optional[torch.Tensor] = None,
        primary_topic_valid_mask: Optional[torch.Tensor] = None,
        tone_valid_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> EconMediaBertOutput:
        encoded = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled = self.dropout(encoded.pooler_output)
        primary_topic_logits = self.primary_topic_classifier(pooled)
        tone_logits = self.tone_classifier(pooled)
        primary_topic_loss = tone_loss = total_loss = None
        if primary_topic_labels is not None:
            safe_labels = primary_topic_labels.long().clamp_min(0)
            per_example = CrossEntropyLoss(reduction="none")(primary_topic_logits, safe_labels)
            mask = primary_topic_valid_mask if primary_topic_valid_mask is not None else (primary_topic_labels >= 0)
            primary_topic_loss = self._masked_mean(per_example, mask)
        if tone_labels is not None:
            safe_labels = tone_labels.long().clamp_min(0)
            per_example = CrossEntropyLoss(reduction="none")(tone_logits, safe_labels)
            mask = tone_valid_mask if tone_valid_mask is not None else (tone_labels >= 0)
            tone_loss = self._masked_mean(per_example, mask)
        components = []
        if primary_topic_loss is not None:
            components.append(self.primary_topic_loss_weight * primary_topic_loss)
        if tone_loss is not None:
            components.append(self.tone_loss_weight * tone_loss)
        if components:
            total_loss = sum(components)
        return EconMediaBertOutput(
            loss=total_loss,
            primary_topic_loss=primary_topic_loss,
            tone_loss=tone_loss,
            primary_topic_logits=primary_topic_logits,
            tone_logits=tone_logits,
        )
