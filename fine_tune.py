import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from tqdm import tqdm
import pandas as pd
import numpy as np

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 8
MAX_LEN = 128
LR = 2e-5
PATIENCE = 3  # early stop after 10 epochs without improvement
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your dataset
df = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\custom_vtuber\\balanced_emotion_dataset.csv")
df['label'] = df['label'].astype(int)
labels = sorted(df['label'].unique())
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

# Tokenizer
tokenizer = BertTokenizer(vocab_file="C:\\Users\\ADMIN\\Desktop\\custom_vtuber\\bert-tiny\\vocab.txt")

# Custom Dataset
class EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label2id[self.labels[idx]], dtype=torch.int64)  # Ensure correct dtype
        }

# Dataset split
dataset = EmotionDataset(df['text'].tolist(), df['label'].tolist())
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Load the configuration from the saved path
config = BertConfig.from_pretrained("C:\\Users\\ADMIN\\Desktop\\custom_vtuber\\bert-tiny\\config.json")

# Convert the config values to avoid `int32` issues
def convert_int32_to_int(config):
    for key, value in config.to_dict().items():
        if isinstance(value, np.int32):  # Check for int32
            config[key] = int(value)  # Convert to Python native int
    return config

config = convert_int32_to_int(config)

# Add num_labels to the configuration
config.num_labels = len(label2id)

# Now load the model with the fixed configuration
model = BertForSequenceClassification(config=config)

# Load the custom weights from the model file
model.load_state_dict(torch.load("C:\\Users\\ADMIN\\Desktop\\custom_vtuber\\bert-tiny\\pytorch_model.bin"), strict=False)
model = model.to(DEVICE)

# Optimizer setup
optimizer = AdamW(model.parameters(), lr=LR)

# Training loop with validation + early stopping
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training")
    total_train_loss = 0

    for batch in train_loop:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_train_loss += loss.item()
        train_loop.set_postfix(train_loss=loss.item())

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "./best_best_emotion_model.bin")
        print("Model improved. Saving checkpoint.")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Save the tokenizer and vocab
tokenizer.save_pretrained("C:\\Users\\ADMIN\\Desktop\\custom_vtuber\\bert-tiny")  # Update with your path where you want to save

# Save the config.json (important for future loading)
config.save_pretrained("C:\\Users\\ADMIN\\Desktop\\custom_vtuber\\bert-tiny")

# Optionally save the model as well if you want to save after the training session ends
torch.save(model.state_dict(), "./best_final_emotion_model.bin")
