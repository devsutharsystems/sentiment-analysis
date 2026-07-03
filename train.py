import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import kagglehub
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    classification_report
)

from text_utils import preprocess_text


def main():
    nltk.download('wordnet')

    path = kagglehub.dataset_download("kazanova/sentiment140")
    print("Path to dataset files:", path)

    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    df = pd.read_csv(
        f"{path}/training.1600000.processed.noemoticon.csv",
        encoding=DATASET_ENCODING,
        names=DATASET_COLUMNS
    )

    data = df[["text", "target"]].copy()
    data.loc[:, "target"] = data["target"].replace(4, 1)

    data_pos = data[data["target"] == 1].iloc[:20000]
    data_neg = data[data["target"] == 0].iloc[:20000]

    dataset = pd.concat([data_pos, data_neg])
    print(dataset)

    # This one line replaces your entire old cleaning block
    dataset.loc[:, "text"] = dataset["text"].apply(preprocess_text)

    X = dataset["text"]
    y = dataset["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=26105111
    )

    vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
    vectoriser.fit(X_train)

    X_train = vectoriser.transform(X_train)
    X_test = vectoriser.transform(X_test)

    plt.figure(figsize=(20, 20))
    wc = WordCloud(max_words=1000, width=1600, height=800, collocations=False).generate(
        " ".join(data_neg["text"])
    )
    plt.imshow(wc)

    BNBmodel = BernoulliNB()
    BNBmodel.fit(X_train, y_train)

    LRmodel = LogisticRegression(max_iter=1000, random_state=42)
    LRmodel.fit(X_train, y_train)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectoriser, f)

    with open("model.pkl", "wb") as f:
        pickle.dump(LRmodel, f)

    print("Model and vectorizer saved successfully!")

    bnb_pred = BNBmodel.predict(X_test)
    lr_pred = LRmodel.predict(X_test)

    bnb_prob = BNBmodel.predict_proba(X_test)[:, 1]
    lr_prob = LRmodel.predict_proba(X_test)[:, 1]

    bnb_accuracy = accuracy_score(y_test, bnb_pred)
    lr_accuracy = accuracy_score(y_test, lr_pred)

    bnb_auc = roc_auc_score(y_test, bnb_prob)
    lr_auc = roc_auc_score(y_test, lr_prob)

    print("=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    print(f"Bernoulli Naive Bayes Accuracy : {bnb_accuracy:.4f}")
    print(f"Logistic Regression Accuracy  : {lr_accuracy:.4f}")

    print(f"Bernoulli Naive Bayes ROC-AUC : {bnb_auc:.4f}")
    print(f"Logistic Regression ROC-AUC  : {lr_auc:.4f}")

    print("\n" + "=" * 50)
    print("BERNOULLI NAIVE BAYES")
    print("=" * 50)
    print(classification_report(y_test, bnb_pred))

    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION")
    print("=" * 50)
    print(classification_report(y_test, lr_pred))

    models = {
    "Bernoulli Naive Bayes": bnb_pred,
    "Logistic Regression": lr_pred
    }

    categories = ["Negative", "Positive"]

    for model_name, predictions in models.items():

        cf_matrix = confusion_matrix(y_test, predictions)

        group_names = [
            "True Neg",
            "False Pos",
            "False Neg",
            "True Pos"
        ]

        group_percentages = [
            "{0:.2%}".format(value)
            for value in cf_matrix.flatten()/np.sum(cf_matrix)
        ]

        labels = [
            f"{v1}\n{v2}"
            for v1, v2 in zip(group_names, group_percentages)
        ]

        labels = np.asarray(labels).reshape(2,2)

        plt.figure(figsize=(6,5))

        sns.heatmap(
            cf_matrix,
            annot=labels,
            cmap="Blues",
            fmt="",
            xticklabels=categories,
            yticklabels=categories
        )

        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        filename = (
            "confusion-matrix-bnb.png"
            if model_name == "Bernoulli Naive Bayes"
            else "confusion-matrix-lr.png"
        )

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

    bnb_fpr, bnb_tpr, _ = roc_curve(y_test, bnb_prob)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)

    plt.figure(figsize=(8,6))

    plt.plot(
        bnb_fpr,
        bnb_tpr,
        label=f"Bernoulli NB (AUC = {bnb_auc:.3f})"
    )

    plt.plot(
        lr_fpr,
        lr_tpr,
        label=f"Logistic Regression (AUC = {lr_auc:.3f})"
    )

    plt.plot([0,1], [0,1], linestyle="--", color="gray", label="Random Guess")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")

    plt.legend()

    plt.grid(True)

    plt.savefig("roc_comparison.png")

    plt.show()


if __name__ == "__main__":
    main()