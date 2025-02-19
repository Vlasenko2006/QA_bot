import json

def prepare_squad_dataset(input_file, output_file):
    with open(input_file, 'r') as f:
        squad_data = json.load(f)

    with open(output_file, 'w') as f:
        for article in squad_data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        answer_text = answer['text']
                        f.write(f"Q: {question}\nA: {answer_text}\n\n")

if __name__ == "__main__":
    # Prepare training and development datasets
    prepare_squad_dataset('./squad/train-v1.1.json', './custom_qa_dataset_train.txt')
    prepare_squad_dataset('./squad/dev-v1.1.json', './custom_qa_dataset_dev.txt')
