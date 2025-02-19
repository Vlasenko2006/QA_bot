# QA_bot



---


# HowTo run

1. Install environment
2. Get the data set
3. Convert it `.txt` format
3. Optionsl: Customize your `slurm` script
4. Run

---

# Environment Installation Instructions

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

#Convert the Data Set to `.txt` Format
Run the script `prepare_squad_dataset.py`

