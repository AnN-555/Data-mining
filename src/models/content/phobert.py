import torch
from transformers import AutoTokenizer, AutoModel

class PhoBERTEncoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.model = AutoModel.from_pretrained("vinai/phobert-base")
        self.model.eval()

    def encode(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # CLS token
        return outputs.last_hidden_state[:, 0, :]
