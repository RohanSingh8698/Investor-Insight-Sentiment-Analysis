import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load the dataset
df = pd.read_csv('data.csv')

# Map sentiment labels to integers
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['sentiment'].map(sentiment_mapping)

# Split the dataset into training, validation, and testing sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(df['news'], df['label'], 
                                                                      random_state=42, 
                                                                      test_size=0.3, 
                                                                      stratify=df['label'])

validation_texts, test_texts, validation_labels, test_labels = train_test_split(temp_texts, temp_labels, 
                                                                                random_state=42, 
                                                                                test_size=0.5, 
                                                                                stratify=temp_labels)

# Tokenization and dataset preparation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_len)
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
val_dataset = NewsDataset(validation_texts, validation_labels, tokenizer)
test_dataset = NewsDataset(test_texts, test_labels, tokenizer)  

# Model initialization
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="steps",  # set save strategy
    save_steps=500,         # Save every 500 steps
    evaluation_strategy="steps",  # set evaluation strategy
    eval_steps=500,         # Evaluate every 500 steps
    load_best_model_at_end=True,
)


# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Start training
trainer.train()




