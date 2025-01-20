import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Ensure GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and fine-tuned model
print("Loading fine-tuned model and tokenizer...")
model_name = "./results"  # Directory where the fine-tuned model is saved
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Define a chatbot interface
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Launching chatbot interface...")
gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="E-Commerce Chatbot",
    description="Ask the chatbot about product inquiries, orders, or returns.",
).launch()
