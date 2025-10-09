#!/usr/bin/env python3

import argparse
import json
import os
from datetime import datetime
import pandas as pd
from datasets import load_dataset
import unicodedata
from nltk.corpus import stopwords
import nltk
import spacy

nltk.download("stopwords", quiet=True)
BIAS_TERMS = {
    "enganoso", "boato", "#fake"
}

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def parse_args():
    """
    Parses command-line arguments for the preprocessing script.

    Returns:
        argparse.Namespace: A namespace containing the following attributes:
            - output_dir (str): Directory where processed data will be saved. Default is "./data/processed".
    """
    p = argparse.ArgumentParser(description="Simple FakeRecogna preprocessing")
    p.add_argument("--output_dir", type=str, default="./data/processed")
    return p.parse_args()

def preprocess(data):
    """
    Preprocesses the input dataset by combining text fields, cleaning, and formatting.
    Args:
        data (pd.DataFrame): A pandas DataFrame containing the columns "Titulo", "Subtitulo", 
            "Noticia", and "Classe". These columns are used to generate a combined text field 
            and a label field.
    Returns:
        pd.DataFrame: A processed DataFrame with two columns:
            - "text": A combined and cleaned text field created from "Titulo", "Subtitulo", 
              and "Noticia".
            - "label": An integer label derived from the "Classe" column.
    """
    data["text"] = data["Titulo"].fillna("") + ". " + data["Subtitulo"].fillna("") + ". " + data["Noticia"].fillna("")
    
    text = data["text"].astype(str)
    
    ''' Removing URLs and links'''
    text = text.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
    
    ''' Removing special characters'''
    text = text.str.replace(r"[^\w\s]", " ", regex=True)
    
    ''' Removing accented characters and converting to lowercase'''
    text = text.str.lower().map(strip_accents)
    
    ''' Removing bias terms'''
    text = text.apply(lambda s: " ".join(w for w in s.split() if w not in BIAS_TERMS))
    
    ''' Removing stop words'''
    stop_words_pt = set(stopwords.words("portuguese"))
    text = text.apply(lambda s: " ".join(w for w in s.split() if w not in stop_words_pt))
    
    ''' Applying lemmatization'''
    nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner", "textcat"])
    text = text.apply(lambda s: " ".join(token.lemma_ for token in nlp(s)))

    ''' Cleaning extra spaces'''
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    data["text"] = text
    
    data = data[["text", "Classe"]].rename(columns={"Classe": "label"})
    dataset = data.dropna(subset=["text", "label"])
    dataset.loc[:, "label"] = dataset["label"].astype(int)
    
    return dataset

def main():
    args = parse_args()
    
    ''' Create output directory if it doesn't exist '''
    os.makedirs(args.output_dir, exist_ok=True) 
    
    ''' Load dataset '''
    dataset = load_dataset("recogna-nlp/FakeRecogna")
    data = dataset["train"]
    df = data.to_pandas()
    
    ''' Preprocess dataset '''
    df_preprocessed = preprocess(df)
    
    X, y = df_preprocessed["text"], df_preprocessed["label"]
    
    pd.DataFrame({"text": X, "label": y}).to_csv(os.path.join(args.output_dir, "dataset.csv"), index=False)
    
    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "num_samples": len(X),
    }    

    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    

if __name__ == "__main__":
    main()