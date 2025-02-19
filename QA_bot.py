#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 22:39:16 2025

Fine-tune a DistilGPT-2 model on a custom QA dataset and implement a question-answer bot.

@author: andrey
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from tqdm import tqdm
import logging
import os
from transformers.utils import logging as transformers_logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_info()

def get_model_size(model_name):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model_path = model.config._name_or_path
    total_size = 0
    for root, dirs, files in os.walk(model_path):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    return total_size / (1024**3)  # Convert to GB

def fine_tune_gpt2(dataset_path, model_name='distilgpt2', output_dir='./fine_tuned_model', num_train_epochs=1, batch_size=2, evaluation_prompt="The future of AI"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    logger.info("Preparing dataset...")
    dataset = TextDataset(tokenizer=tokenizer, file_path=dataset_path, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(output_dir=output_dir, overwrite_output_dir=True, num_train_epochs=num_train_epochs, per_device_train_batch_size=batch_size, report_to="none")

    logger.info("Initializing trainer...")
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=dataset)

    # Generate text for evaluation before training
    logger.info("Generating text before training...")
    input_ids = tokenizer.encode(evaluation_prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1, num_beams=5)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Text before training: ")
    print(generated_text)
    logger.info(f"Generated Text Before Training:\n{generated_text}\n")

    # Training loop with evaluation at each epoch
    logger.info("Starting training...")
    for epoch in tqdm(range(num_train_epochs), desc="Training Epochs"):
        logger.info(f"Starting epoch {epoch + 1}")
        trainer.train()
        logger.info(f"Saving model and tokenizer for epoch {epoch + 1}")
        model.save_pretrained(f"{output_dir}/epoch_{epoch + 1}")
        tokenizer.save_pretrained(f"{output_dir}/epoch_{epoch + 1}")

        # Generate text for evaluation
        logger.info(f"Generating text for epoch {epoch + 1}...")
        input_ids = tokenizer.encode(evaluation_prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model.generate(input_ids, max_length=50, num_return_sequences=3, num_beams=5)
        generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(3)]
        
        logger.info(f"\nEpoch {epoch + 1} Generated Texts:\n")
        print(f"Epoch = {epoch} \n")
        for idx, text in enumerate(generated_texts):
             print(f"Generated Text {idx + 1}:\n{text}\n")

def generate_answer(question, model, tokenizer, max_length=50, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = tokenizer.encode(question, return_tensors='pt').to(device)
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

    generated_answers = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    return generated_answers

if __name__ == "__main__":
    num_train_epochs = 10
    batch_size = 64
    dataset_path = "./squad/custom_qa_dataset_train.txt"  # Path to the prepared QA dataset

    # Verify the dataset size
    dataset_size = os.path.getsize(dataset_path) / (1024**2)  # Convert to MB
    logger.info(f"Total size of the dataset: {dataset_size:.2f} MB")

    fine_tune_gpt2(dataset_path, num_train_epochs=num_train_epochs, batch_size=batch_size)

    # Load the fine-tuned model for QA
    tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model/epoch_100')
    model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model/epoch_100').to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Example QA interaction
    question = "What is the capital of France?"
    answers = generate_answer(question, model, tokenizer)
    for idx, answer in enumerate(answers):
        print(f"Answer {idx + 1}:\n{answer}\n")
