from pathlib import Path
import csv
import numpy as np
import pandas as pd
import pickle
import re
import copy
from collections import Counter
from abc import ABCMeta, abstractmethod
from datasets import load_dataset
import matplotlib.pyplot as plt

############################################################
# TEXT PREPROCESSING
############################################################

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    return text


############################################################
# VOCAB + BAG OF WORDS
############################################################

def build_vocab(texts, max_words=15000):

    counter = Counter()

    for text in texts:
        tokens = preprocess_text(text).split()
        counter.update(tokens)

    most_common = counter.most_common(max_words)

    vocab = {"<pad>":0,"<unk>":1}

    for i,(word,_) in enumerate(most_common,start=2):
        vocab[word] = i

    return vocab


def text_to_bow(text,vocab):

    tokens = preprocess_text(text).split()

    vector = np.zeros(len(vocab))

    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1

    return vector


def build_dataset(df,vocab):

    X = []
    y = []

    for _,row in df.iterrows():

        X.append(text_to_bow(row["Text"],vocab))
        y.append(row["Label"])

    return np.array(X),np.array(y)


############################################################
# LABEL ENCODING
############################################################

def encode_labels(labels):

    unique = sorted(list(set(labels)))

    mapping = {label:i for i,label in enumerate(unique)}

    encoded = np.array([mapping[l] for l in labels])

    return encoded,mapping


def to_one_hot(y,n_classes):

    one_hot = np.zeros((len(y),n_classes))

    for i,label in enumerate(y):
        one_hot[i,label] = 1

    return one_hot

############################################################
# VETOR TF-IDF
############################################################

def compute_idf(texts, vocab):

    N = len(texts)

    df = np.zeros(len(vocab))

    for text in texts:

        tokens = set(preprocess_text(text).split())

        for token in tokens:
            if token in vocab:
                df[vocab[token]] += 1

    idf = np.log((N + 1) / (df + 1)) + 1

    return idf

def text_to_tfidf(text, vocab, idf):

    tokens = preprocess_text(text).split()

    tf = np.zeros(len(vocab))

    for token in tokens:
        if token in vocab:
            tf[vocab[token]] += 1

    if len(tokens) > 0:
        tf = tf / len(tokens)

    tfidf = tf * idf

    norm = np.linalg.norm(tfidf)
    if norm > 0:
        tfidf = tfidf / norm

    return tfidf

def build_dataset_tfidf(df, vocab, idf):

    X = []
    y = []

    for _,row in df.iterrows():

        vec = text_to_tfidf(row["Text"], vocab, idf)

        X.append(vec)
        y.append(row["Label"])

    return np.array(X), np.array(y)

############################################################
# DATASET WRAPPER
############################################################

class Dataset:

    def __init__(self,X,y):
        self.X = X
        self.y = y


############################################################
# OPTIMIZER
############################################################

class Optimizer:

    def __init__(self,learning_rate=0.01,momentum=0.9):

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.retained_gradient = None

    def update(self,w,grad):

        if self.retained_gradient is None:
            self.retained_gradient = np.zeros_like(w)

        self.retained_gradient = (
            self.momentum*self.retained_gradient
            + (1-self.momentum)*grad
        )

        return w - self.learning_rate*self.retained_gradient


class Adam:

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):

        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, grad):

        if self.m is None:

            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        self.t += 1

        self.m = self.beta1 * self.m + (1-self.beta1)*grad
        self.v = self.beta2 * self.v + (1-self.beta2)*(grad**2)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        w = w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return w
############################################################
# LAYERS
############################################################

class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self,input,training):
        pass

    @abstractmethod
    def backward_propagation(self,error):
        pass

    @abstractmethod
    def output_shape(self):
        pass

    @abstractmethod
    def parameters(self):
        pass

    def set_input_shape(self,input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape


############################################################
# DENSE LAYER
############################################################

class DenseLayer(Layer):

    def __init__(self,n_units,input_shape=None):

        self.n_units = n_units
        self._input_shape = input_shape

    def initialize(self,optimizer):

        #self.weights = np.random.randn(self.input_shape()[0],self.n_units)*0.01
        self.weights = np.random.randn(self.input_shape()[0], self.n_units) * np.sqrt(2/self.input_shape()[0])
        self.biases = np.zeros((1,self.n_units))

        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)

    def parameters(self):

        return np.prod(self.weights.shape)+np.prod(self.biases.shape)

    def forward_propagation(self,input,training):

        self.input = input
        self.output = np.dot(input,self.weights)+self.biases

        return self.output

    def backward_propagation(self,output_error):

        input_error = np.dot(output_error,self.weights.T)

        #weights_error = np.dot(self.input.T,output_error)
        lambda_reg = 1e-4
        weights_error = np.dot(self.input.T,output_error) + lambda_reg * self.weights

        bias_error = np.sum(output_error,axis=0,keepdims=True)

        self.weights = self.w_opt.update(self.weights,weights_error)
        self.biases = self.b_opt.update(self.biases,bias_error)

        return input_error

    def output_shape(self):
        return (self.n_units,)


############################################################
# ACTIVATIONS
############################################################

class ReLU(Layer):

    def forward_propagation(self,input,training):

        self.input = input
        return np.maximum(0,input)

    def backward_propagation(self,error):

        return error*(self.input>0)

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return 0


class Softmax(Layer):

    def forward_propagation(self,input,training):

        exp = np.exp(input-np.max(input,axis=1,keepdims=True))

        self.output = exp/np.sum(exp,axis=1,keepdims=True)

        return self.output

    def backward_propagation(self,error):

        return error

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return 0


class Dropout(Layer):

    def __init__(self,rate):
        self.rate = rate

    def forward_propagation(self,input,training):

        if training:

            self.mask = np.random.binomial(1,1-self.rate,input.shape)

            return input*self.mask

        return input

    def backward_propagation(self,error):

        return error*self.mask

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return 0


############################################################
# LOSS
############################################################

class MeanSquaredError:

    def loss(self,y_true,y_pred):
        return np.mean((y_true-y_pred)**2)

    def derivative(self,y_true,y_pred):
        return 2*(y_pred-y_true)/y_true.size

class CategoricalCrossEntropy:

    def loss(self, y_true, y_pred):

        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def derivative(self, y_true, y_pred):

        return (y_pred - y_true) / y_true.shape[0]

############################################################
# METRIC
############################################################

def accuracy(y_true,y_pred):

    y_true = np.argmax(y_true,axis=1)
    y_pred = np.argmax(y_pred,axis=1)

    return np.mean(y_true==y_pred)


############################################################
# NEURAL NETWORK
############################################################

class NeuralNetwork:

    def __init__(
        self,
        epochs=10,
        batch_size=64,
        learning_rate=0.01,
        momentum=0.9,
        verbose=True
    ):

        self.epochs = epochs
        self.batch_size = batch_size
        #self.optimizer = Optimizer(learning_rate,momentum)
        self.optimizer = Adam(learning_rate)

        self.verbose = verbose

        self.layers = []

        #self.loss = MeanSquaredError()
        self.loss = CategoricalCrossEntropy()

    def add(self,layer):

        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())

        if hasattr(layer,"initialize"):
            layer.initialize(self.optimizer)

        self.layers.append(layer)

    def forward(self,X,training):

        output = X

        for layer in self.layers:
            output = layer.forward_propagation(output,training)

        return output

    def backward(self,error):

        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)

    def get_batches(self,X,y):

        n = X.shape[0]

        indices = np.arange(n)
        np.random.shuffle(indices)

        for i in range(0,n,self.batch_size):

            idx = indices[i:i+self.batch_size]

            yield X[idx],y[idx]

    def fit(self,dataset):

        X = dataset.X
        y = dataset.y

        for epoch in range(self.epochs):

            for X_batch,y_batch in self.get_batches(X,y):

                output = self.forward(X_batch,True)

                error = self.loss.derivative(y_batch,output)

                self.backward(error)

            preds = self.forward(X,False)

            acc = accuracy(y,preds)

            if self.verbose:
                print("Epoch",epoch+1,"accuracy:",acc)

    def predict(self,dataset):

        return self.forward(dataset.X,False)


############################################################
# LOGISTIC REGRESSION BASELINE
############################################################

class LogisticRegression:

    def __init__(self,lr=0.01,epochs=100):

        self.lr = lr
        self.epochs = epochs

    def sigmoid(self,z):

        return 1/(1+np.exp(-z))

    def fit(self,X,y):

        n_samples,n_features = X.shape

        self.w = np.zeros((n_features,1))
        self.b = 0

        y = y.reshape(-1,1)

        for _ in range(self.epochs):

            linear = np.dot(X,self.w)+self.b

            y_pred = self.sigmoid(linear)

            dw = (1/n_samples)*np.dot(X.T,(y_pred-y))
            db = (1/n_samples)*np.sum(y_pred-y)

            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self,X):

        linear = np.dot(X,self.w)+self.b

        y_pred = self.sigmoid(linear)

        return (y_pred>0.5).astype(int)

def train_logistic_regression_ovr(X, y, label_map):

    models = {}

    for label, idx in label_map.items():

        y_binary = (y == idx).astype(int)

        model = LogisticRegression(lr=0.05, epochs=200)

        model.fit(X, y_binary)

        models[label] = model

    return models

def predict_logistic_regression_ovr(models, X):

    scores = []

    for label, model in models.items():

        prob = model.sigmoid(np.dot(X, model.w) + model.b)

        scores.append(prob)

    scores = np.hstack(scores)

    return np.argmax(scores, axis=1)
############################################################
# MODEL SAVE / LOAD
############################################################

def save_model(model,vocab,label_map,idf,path):

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path,"wb") as f:

        pickle.dump(
            {
                "model":model,
                "vocab":vocab,
                "idf":idf,
                "labels":label_map
            },
            f
        )


def load_model(path):

    with open(path, "rb") as f:

        data = pickle.load(f)

    return data["model"], data["vocab"], data["labels"], data["idf"]

############################################################
# CONFUSION MATRIX
############################################################
def confusion_matrix(y_true, y_pred, n_classes):

    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    return matrix


def print_confusion_matrix(cm, label_map):

    labels = list(label_map.keys())

    print("\nConfusion Matrix\n")

    print("Predicted →")
    print("True ↓\n")

    print("       ", " ".join(f"{l:>10}" for l in labels))

    for i,row in enumerate(cm):
        print(f"{labels[i]:>7}", " ".join(f"{v:10}" for v in row))

def plot_confusion_matrix(cm, label_map):

    labels = list(label_map.keys())

    fig, ax = plt.subplots(figsize=(8,6))

    im = ax.imshow(cm)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # números dentro das células
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

############################################################
# CSV VALIDATION
############################################################

def read_csv_smart(csv_path):

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)

    delimiter = None
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        pass

    if delimiter:
        df = pd.read_csv(csv_path, sep=delimiter, encoding="utf-8-sig")
    else:
        df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")

    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]

    # Fallback for files that were parsed into a single "ID,Text,Label" column.
    if len(df.columns) == 1 and "," in df.columns[0]:
        df = pd.read_csv(csv_path, sep=",", encoding="utf-8-sig")
        df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]

    return df


def ensure_required_columns(df, required_columns):

    normalized = {str(col).strip().lower(): col for col in df.columns}
    rename_map = {}

    for required in required_columns:
        if required in df.columns:
            continue
        candidate = normalized.get(required.lower())
        if candidate:
            rename_map[candidate] = required

    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found columns: {df.columns.tolist()}"
        )

    return df


def evaluate_csv(model_path,csv_path):

    print("Loading model...")
    model,vocab,label_map,idf = load_model(model_path)

    print("Model loaded.")
    print("Evaluating on CSV...")
    df = read_csv_smart(csv_path)
    df = ensure_required_columns(df,["Text"])

    X = []

    for text in df["Text"]:
        #X.append(text_to_tfidf(text, vocab, idf))
        X = np.array([text_to_tfidf(text, vocab, idf) for text in df["Text"]])

    X = np.array(X)

    dataset = Dataset(X,None)

    preds = model.predict(dataset)

    y_pred = np.argmax(preds, axis=1)

    inv_labels = {v:k for k,v in label_map.items()}

    df["Prediction"] = [inv_labels[p] for p in y_pred]

    # Se houver labels verdadeiras -> calcular matriz
    if "Label" in df.columns:

        known_mask = df["Label"].isin(label_map.keys()).to_numpy()
        unknown_labels = sorted(df.loc[~known_mask, "Label"].astype(str).unique().tolist())

        if unknown_labels:
            print(
                f"\nWarning: skipping {np.sum(~known_mask)} rows with unknown labels "
                f"not present in model: {unknown_labels}"
            )

        if np.any(known_mask):
            y_true = np.array([label_map[l] for l in df.loc[known_mask, "Label"]])
            y_pred_known = y_pred[known_mask]

            cm = confusion_matrix(y_true, y_pred_known, len(label_map))

            print_confusion_matrix(cm, label_map)
            #plot_confusion_matrix(cm, label_map)

            acc = np.mean(y_true == y_pred_known)
            print("\nAccuracy:", acc)
        else:
            print("\nNo known labels found in CSV. Skipping confusion matrix/accuracy.")


    return df


############################################################
# TRAINING PIPELINE
############################################################

def train_model(csv_path, model_path):

    df = read_csv_smart(csv_path)
    df_train = df.groupby("Label").sample(frac=0.8, random_state=42)
    df_test = df.drop(df_train.index)

    df_train = ensure_required_columns(df_train, ["Text", "Label"])
    df_test = ensure_required_columns(df_test, ["Text", "Label"])

    # -----------------------------
    # TF-IDF features
    # -----------------------------
    vocab = build_vocab(df_train["Text"])
    idf = compute_idf(df_train["Text"], vocab)

    X, y = build_dataset_tfidf(df_train, vocab, idf)

    y_encoded, label_map = encode_labels(y)

    # -----------------------------
    # Logistic Regression baseline
    # -----------------------------
    print("\nTraining Logistic Regression baseline...")

    lr_models = train_logistic_regression_ovr(X, y_encoded, label_map)

    X_test, y_test = build_dataset_tfidf(df_test, vocab, idf)
    y_test_encoded = np.array([label_map[l] for l in y_test])

    y_pred_lr = predict_logistic_regression_ovr(lr_models, X_test)

    cm_lr = confusion_matrix(y_test_encoded, y_pred_lr, len(label_map))

    print("\nLogistic Regression confusion matrix:")
    print_confusion_matrix(cm_lr, label_map)
    #plot_confusion_matrix(cm_lr, label_map)

    acc_lr = np.mean(y_test_encoded == y_pred_lr)
    print("\nLogistic Regression Accuracy:", acc_lr)

    # -----------------------------
    # Deep Neural Network
    # -----------------------------
    print("\nTraining DNN...")

    n_classes = len(label_map)

    y_onehot = to_one_hot(y_encoded, n_classes)

    dataset = Dataset(X, y_onehot)

    model = NeuralNetwork()

    model.add(DenseLayer(256, input_shape=(X.shape[1],)))
    model.add(ReLU())
    model.add(Dropout(0.2))

    model.add(DenseLayer(64))
    model.add(ReLU())

    model.add(DenseLayer(n_classes))
    model.add(Softmax())

    model.fit(dataset)

    # -----------------------------
    # DNN evaluation
    # -----------------------------
    y_test_onehot = to_one_hot(y_test_encoded, n_classes)

    test_dataset = Dataset(X_test, y_test_onehot)

    preds = model.predict(test_dataset)

    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(test_dataset.y, axis=1)

    cm = confusion_matrix(y_true, y_pred, len(label_map))

    print("\nDNN confusion matrix:")
    print_confusion_matrix(cm, label_map)
    #plot_confusion_matrix(cm, label_map)

    acc = np.mean(y_true == y_pred)
    print("\nDNN Accuracy:", acc)

    # -----------------------------
    # Save model
    # -----------------------------
    save_model(model, vocab, label_map, idf, model_path)

    return model


############################################################
# LOAD DATASET, TRAIN MODEL, EVALUATE CSV
############################################################

def loadCSV(csv_path, name, df=None):

    dataset = load_dataset(csv_path, name=name)
    print(dataset)
    df_train = dataset["train"].to_pandas()
    if(name == "in_domain"):
        df_test = dataset["test"].to_pandas()

        df_train.rename(columns={"content": "Text", "model": "Label"}, inplace=True)
        df_train.drop(columns=["url"], inplace=True)
        df_train.insert(0, "ID", df_train.index + 1)

        df_test.rename(columns={"content": "Text", "model": "Label"}, inplace=True)
        df_test.drop(columns=["url"], inplace=True)
        df_test.insert(0, "ID", df_test.index + 1)
    else:
        df_test = None
        df_train.rename(columns={"text": "Text", "source": "Label", "id": "ID"}, inplace=True)

    df = pd.concat([df, df_train, df_test], ignore_index=True)

    return df


df1 = loadCSV("MLNTeam-Unical/OpenTuringBench", name="in_domain")
df = loadCSV("artem9k/ai-text-detection-pile", name="", df=df1)

module_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

dataset_path_train = module_path / "data" / "dataset-hf_train.csv"

mapping_classes = {
    "meta-llama": "Meta",
    "qwen": "OpenAI",
    "mistralai": "Mistral",
    "google": "Google",
    "anthropic": "Anthropic",
    "human": "Human",
}

df["Label"] = df["Label"].apply(lambda x: mapping_classes.get(x.split("/")[0].lower(), "Others"))
df = df.sample(20000, random_state=42)
df[["Text","Label"]].to_csv(dataset_path_train, index=False)

print(df)


train_model(
    dataset_path_train,
    f"{module_path}/models/model.pkl"
)


dataset_path_test = module_path / "data" / "dataset-hf_test.csv"
df = df.sample(1000, random_state=42)
df[["Text","Label"]].to_csv(dataset_path_test, index=False)

df = evaluate_csv(
    f"{module_path}/models/model.pkl",
   dataset_path_test
)


