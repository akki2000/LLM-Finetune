
# LLaMA Fine-Tuning for E-Commerce Chatbot

Welcome to the **LLaMA Fine-Tuning Project**! This repository fine-tunes the LLaMA model on an e-commerce customer support dataset to build a domain-specific chatbot capable of handling a variety of customer interactions.

---

## Features
- Fine-tuned **LLaMA 1B** model for conversational AI.
- Dataset: **E-Commerce Customer Support Conversations**.
- Interactive chatbot interface powered by **Gradio**.
- Easy setup and training pipeline using **Hugging Face Transformers**.
- GPU support with CUDA 11.8 for faster training and inference.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Troubleshooting](#troubleshooting)

---

## Introduction
This project fine-tunes the open-source **LLaMA 1B** model to handle e-commerce-specific customer queries. It uses the **NebulaByte/E-Commerce Customer Support Conversations** dataset to train the model for scenarios such as:
- Handling product inquiries.
- Providing order updates.
- Managing return and refund requests.

The final product is an interactive chatbot accessible via a web-based interface built with **Gradio**.

---

## Installation

### Prerequisites
- Python 3.8 or later
- pip (Python package manager)
- An NVIDIA GPU with CUDA 11.8 installed (optional for GPU acceleration)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/llama-finetune-chatbot.git
   cd llama-finetune-chatbot
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install all required libraries using the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify GPU Compatibility**:
   Ensure that PyTorch can access your GPU:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   If this returns `True`, your setup is ready for GPU acceleration.

---

## Usage

### Fine-Tune the Model
Run the fine-tuning script to train the model on the dataset:
```bash
python main.py
```

### Launch the Chatbot
After fine-tuning, launch the Gradio-based chatbot:
```bash
python app.py
```

This will start a web server. Open the URL provided in the terminal (e.g., `http://127.0.0.1:7860`) in your browser to interact with the chatbot.

---

## Troubleshooting

### Common Issues
1. **MemoryError: Not Enough GPU Memory**
   - Reduce the batch size in the `main.py` file:
     ```python
     per_device_train_batch_size = 1
     per_device_eval_batch_size = 1
     ```
   - Enable gradient accumulation to simulate a larger batch size:
     ```python
     gradient_accumulation_steps = 4
     ```

2. **Model Running on CPU Instead of GPU**
   - Ensure you installed the CUDA-enabled PyTorch version:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   - Verify that your GPU is detected:
     ```python
     import torch
     print(torch.cuda.is_available())
     ```

3. **Dataset Not Found Error**
   - Ensure the `NebulaByte/E-Commerce Customer Support Conversations` dataset is accessible.
   - If loading fails, try re-downloading the dataset:
     ```python
     from datasets import load_dataset
     dataset = load_dataset("NebulaByte/E-Commerce_Customer_Support_Conversations")
     ```

4. **RuntimeError: CUDA Out of Memory**
   - Use mixed precision training by enabling `fp16`:
     ```python
     fp16=True
     ```
   - Offload parts of the model to CPU using `accelerate`.

---

## Contributing
We welcome contributions! If you'd like to improve the project:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push your branch and submit a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- [Hugging Face](https://huggingface.co) for their Transformers library.
- [NebulaByte Dataset](https://huggingface.co/datasets/NebulaByte/E-Commerce_Customer_Support_Conversations) for providing the dataset.
- [Meta](https://github.com/facebookresearch/llama) for the LLaMA model.
