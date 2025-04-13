from flask import Flask, request, render_template_string
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_name = './fine_tuned_model/epoch_44'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define a padding token for the tokenizer if it doesn't have one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2LMHeadModel.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Resize the model's embedding layer to account for the new pad_token
model.resize_token_embeddings(len(tokenizer))

@app.route('/')
def index():
    return render_template_string(open('index.html').read())

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = generate_answer(question)
    return render_template_string(open('index.html').read() + f'<div id="answer"><p><strong>Answer:</strong> {answer}</p></div>')

def generate_answer(question, max_length=250, num_return_sequences=1, temperature=0.95, top_k=50, top_p=0.9):
    # Tokenize the input and create the attention mask
    inputs = tokenizer(question, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,  # Pass the attention mask
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id  # Use the new pad_token_id
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Split the generated text into sentences based on ., ?, or !
    sentences = re.split(r'(?<=[.!?])\s+', generated_text)
    answer = ' '.join(sentences[:2])
    return answer

if __name__ == '__main__':
    print("Starting app")
    app.run(host='0.0.0.0', port=5000)