import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_path = "./deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # Don't use float16 on CPU
    device_map={"": "cpu"},
    low_cpu_mem_usage=True
)

# Function for generating responses
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Gradio interface
gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=3, label="Enter your prompt"),
    outputs=gr.Textbox(label="Response"),
    title="DeepSeek 7B Chatbot",
    description="Run DeepSeek 7B locally with a simple UI"
).launch(share=True)

