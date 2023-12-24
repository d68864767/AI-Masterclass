```python
# Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW
from torch.utils.data import Dataset, DataLoader
import torch

# Define constants
MODEL_NAME = 'gpt2'  # Change this to 'gpt4' once it's available
TOKENIZER = GPT2Tokenizer.from_pretrained(MODEL_NAME)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your dataset
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = TOKENIZER.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs["token_type_ids"], dtype=torch.long),
        }

# Define the model
class TextGenerationModel(torch.nn.Module):
    def __init__(self):
        super(TextGenerationModel, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.gpt2(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        return outputs

# Define the training function
def train(model, dataloader, optimizer, device):
    model.train()
    for _, data in enumerate(dataloader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Define the text generation function
def generate_text(model, prompt, max_length=512):
    model.eval()
    inputs = TOKENIZER.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, do_sample=True)
    return TOKENIZER.decode(outputs[0])
```
