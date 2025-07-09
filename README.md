# pLM-ALE

Hello, provided here is some code I used for my thesis. The goal was to fine-tune a protein language model (pLM) to predict impactful mutations in Adaptive Lab Evolution (ALE) data.
Below, you will find some examples on how to run the fine-tuning and what requirements you will need.

## Requirements:
* Set up a Conda environment
* Create a Weights & Biases account: https://wandb.ai/site
* Find a pLM from the HuggingFace platform: https://huggingface.co/ (I used ESM-2 and ProGen2)
* Create a config file for the finetuning

## How to Run:
When you login in to your server on the terminal, you will need to set up the correct conda environment. This is done with:

### requirements.txt Method
```
$ conda create -n "myenv" python=3.10
$ conda activate "myenv"
$ pip install -r requirements.txt
```

### environment.yml Method
```
$ conda env create -f environment.yml
$ conda activate "myenv"
```
As for the fine-tuning runs, you can use:
`python run-finetuning.py`
to start the finetuning process. If you look inside this .py file, I have some default flags that you can change. Most importantly is the `--config` flag. If you want to set up your own
runs, you will need to create a new .yml config file. I have two examples you can base yours off of in `optimized-config.yml` and `finetune_config.yml`. More information on 
what variables you can define in the config file are listed here: https://docs.wandb.ai/guides/track/config/ https://docs.wandb.ai/tutorials/experiments/

If you are successful after this step, you will be asked to provide a Wandb (Weights & Biases) account for logging the fine-tuning. 

## Best Practices
To avoid any suffering for you, here are some tips and insights I can share when I was working on this that I hope will help:
* **Model Size** Pay attention to how big your model is and what your GPU/compute resources are. If you have a lot of data or you want bigger batch sizes to fine-tune faster, your run may crash simply because your memory is not enough to handle your workload.
*  **Weights & Biases** This was an extremely useful tool for this project as it allowed me to identify ways to train these pLMs better. Be sure to follow the dashboard so you can track losses and more. Name your Wandb runs based on what model you are using, the model parameter size, and what kind of fine-tuning technique you want to use for easy bookkeeping (I used Low Rank Adaptation LoRA for fine-tuning).
*  **Dataset** Be sure to follow the standard practice of splitting your dataset up into train, val, and test. I used folders to separate each part, and all my sequences were put into a single `.fasta` file (Ex. `train.fasta`).

There were some other analyses and metrics that I recorded that you can find within the finetune folders. I fine-tuned these models on two datasets; one for E. coli and for another bacteria ADP1. 

If this helps at all please be sure to provide a citation.
