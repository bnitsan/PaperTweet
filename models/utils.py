# partially based on https://github.com/kelvin-jose/T5-Transformer-Lightning , with some additions by me

import pandas as pd
from dataset import CustomDataset
from torch.utils.data import DataLoader
import yaml
import wandb
from pathlib import Path
import os

from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import BLEUScore
rouge_func = ROUGEScore(rouge_keys=('rouge1','rouge2', 'rougeL'))
bleu_func = BLEUScore()


def add_prompt_to_source(df, model_params):
    """Adds a prompt like "summarize: " to the input; useful in a model like T5. Returns modified dataframe"""
    if model_params["ADD_PROMPT_PREFIX"]:
        df[model_params["SOURCE_TITLE"]] = df[model_params["SOURCE_TITLE"]].apply(lambda x: model_params["PREFIX_TO_PROMPT"] + x)
    return df


def add_eos_token_to_target(df, model_params):
    """Adds an end-of-sentence token like </s>. Required in some implementations of T5."""
    df[model_params["TARGET_TITLE"]] = df[model_params["TARGET_TITLE"]].apply(lambda x: x + model_params["EOS_TOKEN"])
    return df


def get_train_dataloaders(path, tokenizer, model_params):
    """Reads train dataframe from path, returns a training dataloader of CustomDataset"""
    df_train = pd.read_csv(path)

    df_train = add_prompt_to_source(df_train, model_params)
    df_train = add_eos_token_to_target(df_train, model_params)
    
    train_ds = CustomDataset(
        df_train,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        model_params["SOURCE_TITLE"],
        model_params["TARGET_TITLE"],
    )
    train_dataloader = DataLoader(train_ds, batch_size=model_params["TRAIN_BATCH_SIZE"], num_workers=model_params["NUM_WORKERS"])
    return train_dataloader, len(df_train)


def get_val_dataloaders(path, tokenizer, model_params):
    """Reads validation dataframe from path, returns a validation dataloader of CustomDataset"""
    df_val = pd.read_csv(path)

    df_val = add_prompt_to_source(df_val, model_params)
    df_val = add_eos_token_to_target(df_val, model_params)

    val_ds = CustomDataset(
        df_val,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        model_params["SOURCE_TITLE"],
        model_params["TARGET_TITLE"],
    )
    val_dataloader = DataLoader(val_ds, batch_size=model_params["VALID_BATCH_SIZE"], num_workers=model_params["NUM_WORKERS"])
    return val_dataloader


def get_test_dataloaders(path, tokenizer, model_params, **inference_params):
    """Reads test dataframe from path, returns a test dataloader of CustomDataset. Uses particular inference_params"""
    df_test = pd.read_csv(path)

    df_test = add_prompt_to_source(df_test, model_params)
    df_test = add_eos_token_to_target(df_test, model_params)

    test_ds = CustomDataset(
        df_test,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        model_params["SOURCE_TITLE"],
        model_params["TARGET_TITLE"],
    )
    test_dataloader = DataLoader(test_ds, **inference_params)
    return test_dataloader


def jaccard(str1, str2):
    """Computes Jaccard score between strings."""
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def rouge_score(pred, target):
    """Computes ROUGE score between strings."""
    rouge_eval = rouge_func(pred, target)
    return rouge_eval['rouge1_fmeasure'], rouge_eval['rouge2_fmeasure'], rouge_eval['rougeL_fmeasure']


def bleu_score(pred, target):
    """Computes BLEU score between strings."""
    return bleu_func(pred, target).item()


def read_yml(file_path):
    """Reads yml files, mostly geared towards config.yml ."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def train_val_test_path(path):
    """We split the train-val-test in advance in a particular convention which we reconstruct here."""
    return path+'_train.csv', path+'_val.csv', path+'_test.csv'


def initialize_wandb(model_params):
    """Initializes the wandb module."""
    with open(str(Path(os.getcwd()).parent.absolute())+'/wandb_key.txt') as key_f:
        wandbkey = key_f.read()
    wandb.login(key=wandbkey)
    wandb.init(project="papertweet", entity="nits")

    wandb.config = {
        "learning_rate": model_params["LEARNING_RATE"],
        "epochs": model_params["TRAIN_EPOCHS"],
        "batch_size": model_params["TRAIN_BATCH_SIZE"]
    }
    return wandb


def get_num_train_steps(model_params, train_df_len):
    """Computes the number of steps per train, given batch size, epochs, number of accumulated batches"""
    return int((train_df_len / model_params["TRAIN_BATCH_SIZE"]) * model_params["TRAIN_EPOCHS"] /
               model_params["TRAIN_BATCH_ACCUM"])
