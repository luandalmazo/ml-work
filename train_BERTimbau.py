#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, Value
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import EarlyStoppingCallback
import torch


def parse_args():
    p = argparse.ArgumentParser(description="BERTimbau + Stratified K-Fold CV on FakeRecogna")
    p.add_argument("--df_path", type=str, default="./data/processed/dataset.csv")
    p.add_argument("--output_dir", type=str, default="./bertimbau_cv")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=256)
    return p.parse_args()


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for a classification model.

    This function calculates the accuracy, precision, recall, and F1-score
    for the given predictions and labels. The precision, recall, and F1-score
    are computed using a weighted average across all classes.

    Args:
        eval_pred (tuple): A tuple containing two elements:
            - preds (numpy.ndarray): The predicted logits or probabilities.
            - labels (numpy.ndarray): The true labels.

    Returns:
        dict: A dictionary containing the following metrics:
            - "accuracy" (float): The accuracy of the predictions.
            - "precision" (float): The weighted precision of the predictions.
            - "recall" (float): The weighted recall of the predictions.
            - "f1" (float): The weighted F1-score of the predictions.
    """
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def tokenize(batch, tokenizer, max_length):
    """
    Tokenizes a batch of text data using the specified tokenizer.

    Args:
        batch (dict): A dictionary containing the text data to be tokenized. 
                      It is expected to have a key "text" with the text data as its value.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the text data.
        max_length (int): The maximum length of the tokenized sequences. 
                          Sequences longer than this will be truncated.

    Returns:
        dict: A dictionary containing the tokenized text data, with padding and truncation applied.
    """
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)


def plot_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix"):
    """
    Plots and saves a confusion matrix as a heatmap.

    Args:
        y_true (list or array-like): Ground truth (true) labels.
        y_pred (list or array-like): Predicted labels.
        output_path (str): File path to save the confusion matrix plot.
        title (str, optional): Title of the plot. Defaults to "Confusion Matrix".

    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout(); plt.savefig(output_path); plt.close()


def plot_loss_curve(log_history, output_path, title="Training vs Validation Loss"):
    """
    Plots the training and validation loss curves over epochs and saves the plot to a file.

    Args:
        log_history (list of dict): A list of dictionaries containing the training and validation 
            loss history. Each dictionary should have keys such as "epoch", "loss", and "eval_loss".
        output_path (str): The file path where the generated plot will be saved.
        title (str, optional): The title of the plot. Defaults to "Training vs Validation Loss".

    """
    logs = pd.DataFrame(log_history)
    train = (logs[logs["loss"].notna()].groupby("epoch", as_index=False)["loss"].last())
    val = (logs[logs["eval_loss"].notna()].groupby("epoch", as_index=False)["eval_loss"].last())
    df = pd.merge(train, val, on="epoch", how="outer").sort_values("epoch")
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["loss"], marker="o", label="Train Loss")
    plt.plot(df["epoch"], df["eval_loss"], marker="s", label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(output_path); plt.close()


def to_hf_dataset(df):
    """
    Converts a pandas DataFrame into a Hugging Face Dataset with specific formatting.

    This function takes a pandas DataFrame, resets its index, and converts it into a Hugging Face Dataset.
    It renames the "label" column to "labels" and ensures that the "labels" column is cast to the "int64" data type.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data to be converted.

    Returns:
        datasets.Dataset: A Hugging Face Dataset with the specified formatting applied.
    """
    ds = Dataset.from_pandas(df.reset_index(drop=True))
    ds = ds.rename_column("label", "labels")
    ds = ds.cast_column("labels", Value("int64"))
    
    return ds


def tokenize_format(ds, tokenizer, max_length):
    """
    Tokenizes and formats a dataset for use with PyTorch.

    This function applies tokenization to the dataset using the provided tokenizer
    and maximum sequence length. It then sets the dataset format to PyTorch tensors
    with the specified columns.

    Args:
        ds (Dataset): The dataset to be tokenized and formatted.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the dataset.
        max_length (int): The maximum sequence length for tokenization.

    Returns:
        Dataset: The tokenized and formatted dataset.
    """
    ds = ds.map(lambda x: tokenize(x, tokenizer, max_length), batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def train_eval_once(train_df, val_df, tokenizer, model_name, args, run_dir, title_suffix=""):
    """
    Trains and evaluates a BERT-based model for sequence classification.

    Args:
        train_df (pd.DataFrame): The training dataset in pandas DataFrame format.
        val_df (pd.DataFrame): The validation dataset in pandas DataFrame format.
        tokenizer (PreTrainedTokenizer): The tokenizer to preprocess the text data.
        model_name (str): The name or path of the pre-trained model to use.
        args (Namespace): A namespace containing training arguments such as seed, batch_size, and epochs.
        run_dir (str): The directory where outputs (e.g., plots, logs) will be saved.
        title_suffix (str, optional): A suffix to append to plot titles. Defaults to an empty string.

    Returns:
        dict: A dictionary containing evaluation metrics for the validation dataset.

    """
    set_seed(args.seed)

    train_ds = tokenize_format(to_hf_dataset(train_df), tokenizer, args.max_length)
    val_ds   = tokenize_format(to_hf_dataset(val_df),   tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    targs = TrainingArguments(
        output_dir=run_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()

    metrics = trainer.evaluate(val_ds)

    os.makedirs(run_dir, exist_ok=True)
    plot_loss_curve(trainer.state.log_history, os.path.join(run_dir, "loss_curve.png"), title=f"Loss {title_suffix}")

    preds = trainer.predict(val_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    plot_confusion_matrix(y_true, y_pred, os.path.join(run_dir, "confusion_matrix.png"), title=f"Confusion {title_suffix}")

    return metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("---" * 20)
    print("BERTimbau CV on FakeRecogna")
    print("Cuda available:", torch.cuda.is_available())
    print("---" * 20)

    model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ''' Loading data '''
    df = pd.read_csv(args.df_path)

    df["label"] = df["label"].astype("int64")

    X = df["text"].values
    y = df["label"].values

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    print("Ok. Starting training...\n")
    fold_metrics = []
    for fold, (train_idx, val_indx) in enumerate(skf.split(X, y), start=1):
        
        print(f"\n--- Fold {fold} - Training ---")
        fold_dir = os.path.join(args.output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        tr_df = df.iloc[train_idx].reset_index(drop=True)
        va_df = df.iloc[val_indx].reset_index(drop=True)

        print(f"[Fold {fold}/{args.k_folds}] train={len(tr_df)} val={len(va_df)}")
        
        m = train_eval_once(tr_df, va_df, tokenizer, model_name, args, run_dir=fold_dir, title_suffix=f"(fold {fold})")

        fold_metrics.append({
            "fold": fold,
            "accuracy": m.get("eval_accuracy", np.nan),
            "precision": m.get("eval_precision", np.nan),
            "recall": m.get("eval_recall", np.nan),
            "f1": m.get("eval_f1", np.nan),
            "loss": m.get("eval_loss", np.nan),
        })
        
        print(f"--- Fold {fold} done ---\n")

    fold_df = pd.DataFrame(fold_metrics).set_index("fold")
    fold_df.to_csv(os.path.join(args.output_dir, "cv_metrics_per_fold.csv"))
    summary = fold_df.agg(["mean", "std"])
    summary.to_csv(os.path.join(args.output_dir, "cv_metrics_mean_std.csv"))
    
    print("---" * 20)
    print("\nCV summary (val):")
    print(summary)
    print("---" * 20)
    
    print("All done.")

if __name__ == "__main__":
    main()
