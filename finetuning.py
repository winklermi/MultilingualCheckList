import torch
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import numpy as np

# Load dataset
tweet_sentiment = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "all") # load all languages

idx2lbl = {idx: label for idx, label in enumerate(tweet_sentiment["train"].features["label"].names)}
lbl2idx = {label: idx for idx, label in enumerate(tweet_sentiment["train"].features["label"].names)}

model_name = "google-bert/bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize dataset
def tokenize_batch(batch):
    return tokenizer(batch["text"], padding="longest", truncation=True)

train_dataset = tweet_sentiment["train"].map(tokenize_batch, batched=True)
val_dataset = tweet_sentiment["validation"].map(tokenize_batch, batched=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(idx2lbl),
    id2label=idx2lbl,
    label2id=lbl2idx,
)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Data collator
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
train_args = TrainingArguments(
    output_dir="./logs/mBERT_sentiment",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=1e-3,
    save_strategy="epoch",
    save_total_limit=1,
    eval_strategy="epoch",
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=len(train_dataset)/4, 
    gradient_accumulation_steps=2,
    seed=42,
    data_seed=42,
    fp16=True,
    load_best_model_at_end=True
)

# Metrics
def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels) # dict(accuracy: score)

# Trainer
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Train and save model
trainer.train()
trainer.save_model("./models/mBERT_sentiment")
