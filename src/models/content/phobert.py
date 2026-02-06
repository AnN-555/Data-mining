import torch
from transformers import AutoTokenizer, AutoModel


class PhoBERTEncoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.model = AutoModel.from_pretrained("vinai/phobert-base")
        self.model.eval()

    def encode(self, texts, batch_size: int = 32, max_length: int = 256):
        """
        Encode a list of texts into CLS embeddings using PhoBERT.
        We process in batches to avoid running out of memory.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            cls_emb = outputs.last_hidden_state[:, 0, :].cpu()
            all_embeddings.append(cls_emb)

        return torch.cat(all_embeddings, dim=0)