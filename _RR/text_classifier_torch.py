import re
from duckdb import df
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from pathlib import Path
from pathlib import Path

############################################
# TEXT PREPROCESSING
############################################

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r"<.*?>","",text)
    text = re.sub(r"\d+","",text)
    text = re.sub(r"[^\w\s]","",text)
    text = re.sub(r"\s+"," ",text).strip()

    return text


############################################
# DATA LOADING
############################################

def load_data():

    ############################################
    # OPENTURINGBENCH
    ############################################

    dataset = load_dataset("MLNTeam-Unical/OpenTuringBench", name="in_domain")

    df_train = dataset["train"].to_pandas().sample(40000, random_state=42)
    df_test = dataset["test"].to_pandas().sample(10000, random_state=42)

    df_train.rename(columns={"content": "Text", "model": "Label"}, inplace=True)
    df_test.rename(columns={"content": "Text", "model": "Label"}, inplace=True)

    ############################################
    # HUMAN DATASET
    ############################################

    dataset_h = load_dataset("artem9k/ai-text-detection-pile")

    df_h = dataset_h["train"].to_pandas()

    df_h = df_h[df_h["source"] == "human"]
    df_h["Text"] = df_h["text"]
    df_h["Label"] = "Human"

    # limitar dataset para não carregar 1M exemplos
    df_h = df_h.sample(10000, random_state=42)

    df_h_train = df_h.sample(frac=0.8, random_state=42)
    df_h_test = df_h.drop(df_h_train.index)

    ############################################
    # ANTHROPIC DATASET
    ############################################

    datasetA = load_dataset("Anthropic/persuasion")

    dfA = datasetA["train"].to_pandas()

    dfA["Text"] = dfA["argument"]
    dfA["Label"] = dfA["source"].apply(
        lambda x: "Anthropic" if str(x).startswith("Claude") else "Human"
    )

    dfA_train = dfA.sample(frac=0.8, random_state=42)
    dfA_test = dfA.drop(dfA_train.index)

    ############################################
    # MERGE DATASETS
    ############################################

    df_train = pd.concat([df_train, df_h_train, dfA_train], ignore_index=True)
    df_test = pd.concat([df_test, df_h_test, dfA_test], ignore_index=True)

    ############################################
    # NORMALIZE LABELS
    ############################################

    mapping_classes = {
        "meta-llama": "Meta",
        "qwen": "OpenAI",
        "mistralai": "Mistral",
        "google": "Google",
        "anthropic": "Anthropic",
        "human": "Human",
    }

    def normalize_label(label):

        key = str(label).split("/")[0].lower()
        return mapping_classes.get(key, label)

    df_train["Label"] = df_train["Label"].apply(normalize_label)
    df_test["Label"] = df_test["Label"].apply(normalize_label)

    ############################################
    # KEEP ONLY TARGET CLASSES
    ############################################

    allowed = ["Meta", "OpenAI", "Mistral", "Google", "Anthropic", "Human"]

    df_train = df_train[df_train["Label"].isin(allowed)]
    df_test = df_test[df_test["Label"].isin(allowed)]

    ############################################
    # BALANCE DATASET
    ############################################

    def balanced_sample(df, n=1000):

        dfs = []

        for label in allowed:

            subset = df[df["Label"] == label]

            if len(subset) >= n:
                dfs.append(subset.sample(n, random_state=42))
            else:
                print(f"Warning: only {len(subset)} samples for {label}")
                dfs.append(subset)

        return pd.concat(dfs, ignore_index=True)

    df_train = balanced_sample(df_train, 1000)
    df_test = balanced_sample(df_test, 1000)

    ############################################
    # INFO
    ############################################

    print("\nTrain distribution:")
    print(df_train["Label"].value_counts())

    print("\nTest distribution:")
    print(df_test["Label"].value_counts())

    return df_train, df_test





############################################
# LABEL ENCODING
############################################

def encode_labels(labels):

    unique = sorted(list(set(labels)))

    # mapping_classes = [
    #     "Meta",
    #     "OpenAI",
    #     "Mistral",
    #     "Google",
    #     "Anthropic",
    #     "Human"
    # ]
    mapping = {label:i for i,label in enumerate(unique)}

    encoded = np.array([mapping[l] for l in labels])

    return encoded,mapping


############################################
# DATASET CLASS
############################################

class TextDataset(Dataset):

    def __init__(self,X,y):

        self.X = X
        self.y = torch.tensor(y,dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):

        x = torch.tensor(self.X[idx].toarray(),dtype=torch.float32).squeeze()
        y = self.y[idx]

        return x,y


############################################
# TF-IDF WITH NGRAMS
############################################

def build_vectorizer():

    return TfidfVectorizer(
        max_features=12000,
        ngram_range=(1,2),
        min_df=5,
        stop_words="english"
    )


############################################
# MODELS
############################################

class LogisticRegression(nn.Module):

    def __init__(self,input_dim,n_classes):

        super().__init__()

        self.linear = nn.Linear(input_dim,n_classes)

    def forward(self,x):

        return self.linear(x)


class DNNClassifier(nn.Module):

    def __init__(self,input_dim,n_classes):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128,n_classes)

        )

    def forward(self,x):

        return self.net(x)

class LinearClassifier(nn.Module):

    def __init__(self,input_dim,n_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim,n_classes)

    def forward(self,x):
        return self.fc(x)
    
############################################
# EMBEDDING + LSTM
############################################

class LSTMClassifier(nn.Module):

    def __init__(self,input_dim,embed_dim=128,hidden_dim=128,n_classes=6):

        super().__init__()

        self.embedding = nn.Linear(input_dim,embed_dim)

        self.lstm = nn.LSTM(embed_dim,hidden_dim,batch_first=True)

        self.fc = nn.Linear(hidden_dim,n_classes)

    def forward(self,x):

        x = self.embedding(x)

        x = x.unsqueeze(1)

        output,(h,c) = self.lstm(x)

        return self.fc(h[-1])


############################################
# EMBEDDING + GRU
############################################

class GRUClassifier(nn.Module):

    def __init__(self,input_dim,embed_dim=128,hidden_dim=128,n_classes=6):

        super().__init__()

        self.embedding = nn.Linear(input_dim,embed_dim)

        self.gru = nn.GRU(embed_dim,hidden_dim,batch_first=True)

        self.fc = nn.Linear(hidden_dim,n_classes)

    def forward(self,x):

        x = self.embedding(x)

        x = x.unsqueeze(1)

        output,h = self.gru(x)

        return self.fc(h[-1])


############################################
# TRAIN FUNCTION
############################################

def train_epoch(model,loader,criterion,optimizer,device):

    model.train()

    total_loss = 0

    for X,y in loader:

        X,y = X.to(device),y.to(device)

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs,y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(loader)


############################################
# EVALUATE
############################################

def evaluate(model,loader,device):

    model.eval()

    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for X,y in loader:

            X,y = X.to(device),y.to(device)

            outputs = model(X)

            preds = torch.argmax(outputs,dim=1)

            correct += (preds==y).sum().item()

            total += y.size(0)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return correct/total,np.array(y_true),np.array(y_pred)


############################################
# CONFUSION MATRIX
############################################

def confusion_matrix(y_true,y_pred,n_classes):

    cm = np.zeros((n_classes,n_classes),dtype=int)

    for t,p in zip(y_true,y_pred):

        cm[t][p]+=1

    return cm


############################################
# EARLY STOPPING
############################################

class EarlyStopping:

    def __init__(self,patience=3):

        self.patience = patience
        self.best = None
        self.counter = 0

    def step(self,metric):

        if self.best is None or metric > self.best:

            self.best = metric
            self.counter = 0

            return False

        else:

            self.counter += 1

            return self.counter >= self.patience


############################################
# TRAIN LOOP
############################################

def train_model(model,train_loader,test_loader,device,epochs=20):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    writer = SummaryWriter()

    early_stop = EarlyStopping()

    for epoch in range(epochs):

        loss = train_epoch(model,train_loader,criterion,optimizer,device)

        acc,_,_ = evaluate(model,test_loader,device)

        writer.add_scalar("Loss/train",loss,epoch)
        writer.add_scalar("Accuracy/test",acc,epoch)

        print(f"Epoch {epoch+1} | loss {loss:.4f} | acc {acc:.4f}")

        if early_stop.step(acc):

            print("Early stopping triggered")
            break

    return model


############################################
# MAIN
############################################

def main():

    df_train,df_test = load_data()

    df_train["Text"] = df_train["Text"].apply(preprocess_text)
    df_test["Text"] = df_test["Text"].apply(preprocess_text)

    vectorizer = build_vectorizer()

    X_train = vectorizer.fit_transform(df_train["Text"])
    X_test = vectorizer.transform(df_test["Text"])


    y_train,label_map = encode_labels(df_train["Label"])
    y_test = np.array([label_map[l] for l in df_test["Label"]])

    n_classes = len(label_map)

    train_dataset = TextDataset(X_train,y_train)
    test_dataset = TextDataset(X_test,y_test)

    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################################
    # MODEL CHOICE
    ############################################

    #model = GRUClassifier(X_train.shape[1],n_classes=n_classes).to(device)
    #model = LinearClassifier(X_train.shape[1],n_classes=n_classes).to(device)
    #model = DNNClassifier(X_train.shape[1],n_classes=n_classes).to(device)
    #model = LSTMClassifier(X_train.shape[1],n_classes=n_classes).to(device)
    model = LogisticRegression(X_train.shape[1],n_classes=n_classes).to(device)

    model = train_model(model,train_loader,test_loader,device)

    

    module_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    dataset_path_train = module_path / "models"
    Path(dataset_path_train).mkdir(parents=True, exist_ok=True)

    torch.save({
        #"model_type": "gru",
        "model_type": "logistic_regression",
        "model_state": model.state_dict(),
        "label_map": label_map,
        "vectorizer": vectorizer
    }, f"{dataset_path_train}/model.pth")

    print("Model saved.")

    acc,y_true,y_pred = evaluate(model,test_loader,device)

    print("\nFinal accuracy:",acc)

    cm = confusion_matrix(y_true,y_pred,n_classes)

    print(cm)


if __name__ == "__main__":
    main()