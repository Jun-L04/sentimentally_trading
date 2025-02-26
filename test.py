import os
import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# clear unused memory in GPU
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
# avoid fragmentation (out of memory error)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# step 1: load the opt model and tokenizer
model_name = "facebook/opt-2.7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2)

# Prepare model for low-rank adaptation
model = prepare_model_for_kbit_training(model)


# Step 2: Load and format your dataset
def load_custom_dataset(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    headlines = [item['headline'] for item in data]
    labels = [item['label'] for item in data]
    return Dataset.from_dict({'text': headlines, 'label': labels})


dataset = load_custom_dataset('training_data.json')


# Step 3: Tokenize the dataset
# include labels for loss
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)


# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir="./opt-finetuned",       # Directory to save the fine-tuned model
    eval_strategy='no',       # Disable evaluation, we don't have evaluation dataset
    learning_rate=5e-5,                  # Learning rate
    per_device_train_batch_size=6,       # Batch size per GPU
    num_train_epochs=50,                  # Number of training epochs
    logging_dir="./logs",                # Directory for logs
    logging_steps=100,                    # Log frequency
    save_strategy="epoch",               # Save checkpoint after each epoch
    fp16=True,                           # Enable mixed precision training
)


# Step 5: Initialize the Trainer
# Trainer() automatically uses GPU if necessary
# libraries are installed and GPU is available
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)


# Step 6: Fine-tune the model
print('Training model...')
trainer.train()


# Step 7: Save the fine-tuned model\
print('Saving model...')
trainer.save_model("./opt-finetuned")
tokenizer.save_pretrained("./opt-finetuned")
