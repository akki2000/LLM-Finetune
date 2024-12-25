import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from accelerate import Accelerator

# Set environment variables to optimize CUDA memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Ensure GPU is used if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and model with CPU offloading
print("Loading model and tokenizer...")
model_name = "unsloth/Llama-3.2-1B"  # Replace with your model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_cache=False,        # Disable caching for gradient checkpointing compatibility
    device_map="balanced_low_0",  # Offload parts of the model to CPU
    offload_folder="./offload",  # Temporary directory for offloaded layers
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Enable gradient checkpointing
print("Enabling gradient checkpointing...")
model.gradient_checkpointing_enable()

# Load and preprocess the dataset
print("Loading and preprocessing dataset...")
dataset = load_dataset("NebulaByte/E-Commerce_Customer_Support_Conversations")
dataset = dataset["train"].shuffle(seed=42).select(range(1000))  # Use only a subset of the dataset

def preprocess_function(examples):
    inputs = tokenizer(examples["conversation"], truncation=True, padding="max_length", max_length=32)  # Shorter length
    inputs["labels"] = inputs["input_ids"].copy()  # Labels required for loss calculation
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split dataset into training and validation
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
tokenized_dataset["validation"] = tokenized_dataset.pop("test")

# Define training arguments with memory optimization
training_args = TrainingArguments(
    output_dir="./results",               # Save directory for model and checkpoints
    evaluation_strategy="epoch",         # Evaluate after every epoch
    learning_rate=5e-5,                  # Learning rate
    per_device_train_batch_size=1,       # Reduce batch size to save memory
    gradient_accumulation_steps=2,       # Simulate larger batch size
    num_train_epochs=3,                  # Number of epochs
    save_total_limit=2,                  # Limit number of saved checkpoints
    fp16=True,                           # Enable mixed precision for faster training
    logging_dir="./logs",                # Directory for logs
    logging_steps=10,                    # Log every 10 steps
    deepspeed="./ds_config.json",        # Enable Deepspeed for advanced memory optimization
)

# Initialize Accelerator for CPU offloading
accelerator = Accelerator(cpu_offload=True)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    accelerator=accelerator,  # Enable CPU offloading
)

# Main function
if __name__ == "__main__":
    # Clear GPU cache to free memory
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=device, abbreviated=True))

    print("Starting fine-tuning...")
    try:
        trainer.train()
        print("Fine-tuning completed successfully!")
    except RuntimeError as e:
        print("RuntimeError:", e)
        print("Try reducing dataset size, sequence length, or enabling more aggressive optimizations.")
