import re
import string
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer

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
    st = PorterStemmer()
    return [st.stem(word) for word in data]

def lemmatizer_text(data):
    lm = WordNetLemmatizer()
    return [lm.lemmatize(word) for word in data]

def preprocess_text(text: str) -> str:
    """Full pipeline: raw text -> cleaned string ready for vectorizer"""
    text = text.lower()
    text = cleaning_stopwords(text)
    text = cleaning_punctuations(text, string.punctuation)
    text = cleaning_repeating_char(text)
    text = cleaning_URLs(text)
    text = cleaning_numbers(text)

    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    tokens = stemming_text(tokens)
    tokens = lemmatizer_text(tokens)

    return " ".join(tokens)