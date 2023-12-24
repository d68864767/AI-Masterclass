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
class MyDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return TOKENIZER.encode(text)

# Replace this with your own data
texts = ["Hello, world!", "Machine learning is fun.", "I love programming."]
dataset = MyDataset(texts)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Load the model
config = GPT2Config.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel(config)
model.to(DEVICE)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):  # Number of epochs
    for batch in dataloader:
        # Move data to device
        batch = torch.stack(batch).to(DEVICE)

        # Forward pass
        outputs = model(batch, labels=batch)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'Epoch: {epoch}, Loss: {loss.item()}')
```
