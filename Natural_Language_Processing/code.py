```python
# Import necessary libraries
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

# Define constants
MODEL_NAME = 'bert-base-uncased'
TOKENIZER = BertTokenizer.from_pretrained(MODEL_NAME)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your dataset
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        inputs = TOKENIZER.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define your model
class TextClassificationModel(torch.nn.Module):
    def __init__(self):
        super(TextClassificationModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(MODEL_NAME)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return F.softmax(outputs.logits, dim=-1)

# Define your training function
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```
