# Fine-tuning BERTimbau para Detecção de Fake News (FakeRecogna)

Script para treinar e avaliar o modelo **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) usando **validação cruzada estratificada (k-fold)** no dataset **FakeRecogna**.

---

## 📁 Estrutura
- `preprocess.py` → faz o pré-processamento (limpeza, lematização e remoção de stopwords).
- `train_BERTimbau_CV.py` → realiza o fine-tuning do BERTimbau com validação cruzada.
- `data/processed/dataset.csv` → dataset final após o pré-processamento.
- `bertimbau_output/` → métricas, curvas de loss e matrizes de confusão por fold.

---

## ⚙️ Pré-requisitos
```bash
python -m spacy download pt_core_news_sm
pip install -r requirements.txt
```

## Pré-processando
```python preprocess.py --output_dir ./data/processed```

## Treinando
```python train_BERTimbau_CV.py --df_path ./data/processed/dataset.csv --output_dir ./bertimbau_output --k_folds 5```