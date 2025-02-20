from flask import Flask, request, render_template_string
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_name = './fine_tuned_model/epoch_100'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

@app.route('/')
def index():
    return render_template_string(open('index.html').read())

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = generate_answer(question)
    return render_template_string(open('index.html').read() + f'<div id="answer"><p><strong>Answer:</strong> {answer[0]}</p></div>')

def generate_answer(question, max_length=50, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.9):
    input_ids = tokenizer.encode(question, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    return [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
