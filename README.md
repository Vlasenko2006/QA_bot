# QA_bot

# Question-Answer Bot using GPT-2

## Overview

This project demonstrates the creation of a Question-Answer (QA) bot using a fine-tuned GPT-2 model. The bot is capable of answering questions based on a custom dataset. The code showcases several important skills relevant for a Data Scientist position, including data handling, model fine-tuning, and practical application development.

## Features

- **Data Handling and Preprocessing**: Loading and preparing datasets (e.g., SQuAD dataset).
- **Model Fine-Tuning**: Fine-tuning a pre-trained language model (DistilGPT-2) on a custom dataset.
- **Use of Transformer Models**: Utilizing advanced NLP models from the `transformers` library.
- **Logging and Monitoring**: Setting up logging to track the training process.
- **Model Evaluation**: Evaluating the model's performance by generating text before and after training.
- **Practical Application**: Implementing a real-world application (QA bot).





### Example Interaction

After running the script, you can interact with the QA bot by asking questions. For example:

```markdown
Question: "What is the capital of France?"
Answer 1: "The capital of France is Paris."
Answer 2: "Paris is the capital of France."
Answer 3: "France's capital city is Paris."
```


## Getting Started

## Overview

1. Install environment
2. Get the data set
3. Convert it `.txt` format
3. Optionsl: Customize your `slurm` script
4. Run


---

# Install environment

### Prerequisites

- Python 3.8 or higher
- `pip` for package management



## Pip Insatllation

Follow the steps below to set up and install the dependencies for the QA bot using a virtual environment and pip.


### Step 1: Make the Script Executable

Make the script executable by running the following command in your terminal:

```sh
chmod +x qa_env_pip.yaml
```

### Step 2: Run the Setup Script

Run the script to create the virtual environment and install the required packages:

```sh
./qa_env_pip.yaml
```

### Step 3: Activate the Virtual Environment

After the setup is complete, activate the virtual environment using the following command:

```sh
source gpt2_finetuning_env/bin/activate
```


You should see the versions of the installed packages printed out without any errors.


## Ananconda Installation

Follow the steps below to set up and install the dependencies for the QA bot.

## Step 1: Create a Virtual Environment

First, ensure you have `conda` installed on your machine. You can download and install Anaconda from [here](https://www.anaconda.com/products/distribution).

Open a terminal or command prompt and create a new conda environment using the provided `.yaml` file:

```sh
conda env create -f environment.yaml
```

This will create a new environment named `qa_bot_env` and install all the necessary dependencies.

## Step 2: Activate the Virtual Environment

Activate the newly created conda environment:

```sh
conda activate qa_bot_env
```

# Getting Data Sets


## Via `wget`
Copy the line and run it your `bash` terminal: 

`wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json`
and this one:
`wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json`

## Via Python Script
Run the script `download_dataset.py`. It will create a directory `squad` and download `.json` datasets there.

# Convert the Data Set to `.txt` Format
Run the script `prepare_squad_dataset.py`


# Run the code

## In a Command Line (not recomended, just for debugging)
Activate your virtual environment as discussed in section **Environment Installation Instructions** and run the code:
### For Anaconda Environment:
`python QA_bot.py`
### For Pip Environment: 
`qa_env_pip/bin/python3.10 QA_bot`
## As Slurm job:
Set all necessary variables in your `run.bash` script ( see more [here](https://slurm.schedmd.com/quickstart.html)   ) and run:
`sbatch run.bash`

