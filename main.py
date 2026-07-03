import re
import numpy as np
import pandas as pd
import string
import seaborn as sns
import kagglehub
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


stopwordlist = [
    "a",
    "about",
    "above",
    "after",
    "again",
    "ain",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "by",
    "can",
    "d",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "ll",
    "m",
    "ma",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "now",
    "o",
    "of",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "own",
    "re",
    "s",
    "same",
    "she",
    "shes",
    "should",
    "shouldve",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "thatll",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "ve",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won",
    "y",
    "you",
    "youd",
    "youll",
    "youre",
    "youve",
    "your",
    "yours",
    "yourself",
    "yourselves",
]
STOPWORDS = set(stopwordlist)

def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def cleaning_punctuations(text, punctuation_list):
    translator = str.maketrans("", "", punctuation_list)
    return text.translate(translator)


def cleaning_repeating_char(text):
    return re.sub(r"(.)1+", r"1", text)


def cleaning_URLs(data):
    return re.sub("((www.[^s]+)|(https?://[^s]+))", " ", data)


def cleaning_numbers(data):
    return re.sub("[0-9]+", "", data)


def stemming_text(data):
    st = nltk.PorterStemmer()
    text = [st.stem(word) for word in data]
    return data


def lemmatizer_text(data):
    lm = nltk.WordNetLemmatizer()
    text = [lm.lemmatize(word) for word in data]
    return data



def main():
    nltk.download('wordnet')
    
    path = kagglehub.dataset_download("kazanova/sentiment140")
    print("Path to dataset files:", path)

    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    df = pd.read_csv(f"{path}/training.1600000.processed.noemoticon.csv", encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
    df.sample(5)
    df.head()

    # ax = (df.groupby("target").count().plot(kind="bar", title="Distribution of data", legend=False))
    # ax.set_xticklabels(["Negative", "Positive"], rotation=0)
    # text, sentiment = list(df["text"]), list(df["target"])
    # sns.countplot(x="target", data=df)


    # prepare for the data
    data = df[["text", "target"]]
    data["target"] = data["target"].replace(4, 1)

    data_pos = data[data["target"] == 1]
    data_neg = data[data["target"] == 0]

    data_pos = data_pos.iloc[: int(20000)]
    data_neg = data_neg.iloc[: int(20000)]

    dataset = pd.concat([data_pos, data_neg])
    print(dataset)

    dataset["text"] = dataset["text"].str.lower()
    dataset["text"] = dataset["text"].apply(lambda text: cleaning_stopwords(text))
    punctuation_list = string.punctuation
    dataset["text"] = dataset["text"].apply(lambda x: cleaning_punctuations(text=x, punctuation_list=punctuation_list))
    dataset["text"] = dataset["text"].apply(lambda x: cleaning_repeating_char(x))
    dataset["text"] = dataset["text"].apply(lambda x: cleaning_URLs(x))
    dataset["text"] = dataset["text"].apply(lambda x: cleaning_numbers(x))


    tokenizer = RegexpTokenizer(r"w+")
    dataset["text"] = dataset["text"].apply(tokenizer.tokenize)
    dataset["text"] = dataset["text"].apply(lambda x: stemming_text(x))
    dataset["text"] = dataset["text"].apply(lambda x: lemmatizer_text(x))


    X = data.text
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=26105111
    )

    vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
    vectoriser.fit(X_train)

    X_train = vectoriser.transform(X_train)
    X_test = vectoriser.transform(X_test)


    # Step-8: Function for Model Evaluation
    # After training the model, we then apply the evaluation measures to check how the model is performing. Accordingly, we use the following evaluation parameters to check the performance of the models respectively:

    plt.figure(figsize=(20, 20))
    wc = WordCloud(max_words=1000, width=1600, height=800, collocations=False).generate(
        " ".join(data_neg)
    )
    plt.imshow(wc)


    BNBmodel = BernoulliNB()
    BNBmodel.fit(X_train, y_train)
    y_pred = BNBmodel.predict(X_test)

    cf_matrix = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)


    categories = ["Negative", "Positive"]
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_percentages = [
        "{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)
    ]
    labels = [f"{v1}n{v2}" for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(
        cf_matrix,
        annot=labels,
        cmap="Blues",
        fmt="",
        xticklabels=categories,
        yticklabels=categories,
    )
    plt.xlabel("Predicted values", fontdict={"size": 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={"size": 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={"size": 18}, pad=20)


    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=1, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC CURVE")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()