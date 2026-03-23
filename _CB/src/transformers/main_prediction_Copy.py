# Este main_prediction.py só contém o código para o transformer de raíz, sem o BERT.
# O código para o BERT está em main_prediction.py

import numpy as np
from pathlib import Path

import torch
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# from transformers.models.dnn import DNNClassifier
from prepare.feature import encode_labels, preprocess_text, preprocess_text_clean, build_vectorizer, build_handcrafted_matrix, standardize_train_test
from prepare.dataset import TextDataset, get_datasets
from models.transformers import TransformerClassifier
# from transformers.prepare.model import evaluate, train_model
# from transformers.prepare.dataset import TextDataset, get_datasets
# from transformers.prepare.feature import preprocess_text, build_vectorizer, encode_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    #############################################################################################################################

    module_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

    checkpoint = torch.load(f"{module_path}/trained_models/transformer.pth", map_location=device, weights_only=False)

    input_dim = checkpoint["input_dim"]
    n_classes = len(checkpoint["label_map"])
    seq_len_model = checkpoint["seq_len"]

    model = TransformerClassifier(input_dim, n_classes, seq_len=seq_len_model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    vectorizer = checkpoint["vectorizer"]
    label_map = checkpoint["label_map"]

    idx_to_label = {v: k for k, v in label_map.items()}

    #############################################################################################################################

    # Ler CSV
    import pandas as pd

    df_new = pd.read_csv(f"{module_path}/data/subm2.csv", sep=";")

    # Preprocessamento (igual ao treino)

    texts = df_new["Text"].tolist()

    texts_clean = [preprocess_text(t) for t in texts]
    clean_light = [preprocess_text_clean(t) for t in texts]

    #############################################################################################################################

    # TF-IDF
    X_tfidf = vectorizer.transform(texts_clean)

    # Handcrafted features
    X_hand, _ = build_handcrafted_matrix(texts, clean_light)

    mean = checkpoint["hand_mean"]
    std = checkpoint["hand_std"]

    X_hand = (X_hand - mean) / std

    # Combinar + normalizar
    import numpy as np

    X = np.hstack([X_tfidf.toarray(), X_hand])

    # PSEUDO-SEQUÊNCIA (igual ao treino)
    seq_len = checkpoint["seq_len"]

    pad_size = (seq_len - (X.shape[1] % seq_len)) % seq_len
    if pad_size > 0:
        X = np.hstack([X, np.zeros((X.shape[0], pad_size))])

    embed_dim = X.shape[1] // seq_len
    X = X.reshape(-1, seq_len, embed_dim)

    # Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Prever
    with torch.no_grad():
        logits = model(X_tensor)
        preds = logits.argmax(dim=1).cpu().numpy()

    # Converter labels
    pred_labels = [idx_to_label[p] for p in preds]

    # Guardar resultados
    df_new["Prediction"] = pred_labels

    df_new.to_csv(f"{module_path}/data/predictions2.csv", index=False)

    print("Predictions saved!")

    #############################################################################################################################
    
    print("Evaluation/Prediction complete.")

if __name__ == "__main__":
    main()   
