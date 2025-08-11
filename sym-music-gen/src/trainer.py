import transformers
import torch


class MIREXTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ce = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        labels = inputs.get("labels", inputs["input_ids"])
        if "labels" in inputs:
            inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

        # shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
        shift_labels = labels[:, 1:].contiguous()  # [B, T-1]

        attn = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
        shift_labels = shift_labels.masked_fill(attn[:, 1:] == 0, -100)

        # CE expects [N, C] and [N]
        loss = self.ce(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss
