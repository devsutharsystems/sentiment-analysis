# Social Media Sentiment Analysis API

An end-to-end **Natural Language Processing (NLP)** project that classifies social media text into **Positive** or **Negative** sentiment using machine learning.

The project covers the complete ML lifecycle — text preprocessing, feature engineering, model comparison, REST API development, Docker containerization, and deployment on **AWS EC2**.

Live demo: `http://51.21.180.81:8000/docs`

---

## Table of Contents

- [Overview](#overview)
- [Project Demonstration](#project-demonstration)
- [Key Features](#key-features)
- [Business Use Case](#business-use-case)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation](#model-evaluation)
- [Technology Stack](#technology-stack)
- [API Endpoints](#api-endpoints)
- [Docker Deployment](#docker-deployment)
- [AWS EC2 Deployment](#aws-ec2-deployment)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Results Summary](#results-summary)
- [Future Improvements](#future-improvements)

---

## Overview

Social media platforms generate millions of opinions every day about products, services, companies, and public events. Manually reading through this volume of text is slow and doesn't scale.

This project builds a complete sentiment analysis pipeline that automatically classifies social media posts as **Positive** or **Negative**, then serves predictions through a deployed REST API.

Rather than training a single model, this project trains and compares **Bernoulli Naive Bayes** and **Logistic Regression**, evaluates both on multiple metrics, and deploys the better-performing model to production.

---

## Project Demonstration

### API Running on AWS EC2

<img width="1128" height="318" alt="image" src="https://github.com/user-attachments/assets/99de3760-ae8e-4e13-93a0-d3ad114982ea" />

### Health Check (Server-Side)

<img width="1554" height="114" alt="image" src="https://github.com/user-attachments/assets/75438890-142d-4b8d-9ab5-6d3358a424bf" />

### Interactive Swagger Documentation

<img width="1290" height="534" alt="image" src="https://github.com/user-attachments/assets/f4185c7b-2ba4-4a4f-9217-f9834373d97e" />

### Sentiment Prediction Example

<img width="924" height="1302" alt="image" src="https://github.com/user-attachments/assets/cf84b000-6677-4be2-abc8-44e81b452a7c" />

---

## Key Features

- End-to-end NLP pipeline
- Text preprocessing (stopwords, punctuation, URLs, numbers, stemming, lemmatization)
- TF-IDF vectorization (unigrams + bigrams)
- Two-model comparison: Bernoulli Naive Bayes vs. Logistic Regression
- Accuracy, ROC-AUC, and confusion matrix evaluation for both models
- REST API built with FastAPI, documented via Swagger/OpenAPI
- Dockerized application, published to Docker Hub
- Live deployment on AWS EC2

---

## Business Use Case

The model is trained on the **Sentiment140** benchmark dataset, but the underlying workflow generalizes directly to real business problems:

- Brand reputation monitoring
- Customer feedback and review analysis
- Social media monitoring
- Marketing campaign sentiment tracking
- Opinion mining at scale

The same pipeline — preprocessing → TF-IDF → classifier → API — can be retrained on customer reviews, support tickets, or survey responses with minimal changes.

---

## Dataset

**Source:** [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) (via `kagglehub`)

| Property | Value |
|---|---|
| Total tweets in source dataset | 1.6 million |
| Task | Binary sentiment classification |
| Labels used | 0 = Negative, 1 = Positive (originally 4, remapped) |
| Training subset | 20,000 positive + 20,000 negative (40,000 total, balanced) |

A balanced subset was used to keep training and iteration fast during development.

---

## Project Workflow

```text
             Social Media Text
                     │
                     ▼
            Text Preprocessing
                     │
                     ▼
           TF-IDF Vectorization
                     │
                     ▼
        Machine Learning Models
   ┌────────────────────────────┐
   │  Bernoulli Naive Bayes     │
   │  Logistic Regression       │
   └────────────────────────────┘
                     │
                     ▼
             Model Comparison
                     │
                     ▼
           Best Model Selected
                     │
                     ▼
             FastAPI REST API
                     │
                     ▼
             Docker Container
                     │
                     ▼
            AWS EC2 Deployment
```

---

## Data Preprocessing

Implemented in `text_utils.py` and shared between training and inference so predictions are always preprocessed identically to training data:

- Lowercasing
- Stopword removal
- URL removal
- Numeric character removal
- Punctuation removal
- Tokenization (`RegexpTokenizer`)
- Stemming (`PorterStemmer`)
- Lemmatization (`WordNetLemmatizer`)

---

## Feature Engineering

Cleaned text is vectorized using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

| Setting | Value |
|---|---|
| N-gram range | Unigrams + Bigrams (1, 2) |
| Max features | 500,000 |

TF-IDF weights informative, distinctive terms more heavily than frequently occurring, low-signal words.

---

## Machine Learning Models

| Model | Role |
|---|---|
| Bernoulli Naive Bayes | Baseline classifier |
| Logistic Regression | Final production model |

Both models were trained on identical training data and evaluated on the same held-out test split for a fair comparison.

---

## Model Evaluation

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Bernoulli Naive Bayes | **75.75%** | **0.829** |
| Logistic Regression | **76.60%** | **0.840** |

### Bernoulli Naive Bayes — Confusion Matrix

<img width="1170" height="982" alt="image" src="https://github.com/user-attachments/assets/731b3dd9-21e8-45f1-ac3f-805105478c07" />

### Logistic Regression — Confusion Matrix

<img width="1132" height="972" alt="image" src="https://github.com/user-attachments/assets/19dcf84e-91ee-4835-8cd6-cb01a0b73aea" />

### ROC Curve Comparison

<img width="1250" height="952" alt="image" src="https://github.com/user-attachments/assets/25647b52-f414-4042-9268-6b1b4170e23e" />

### Final Model Selection

**Logistic Regression** achieved the higher accuracy and ROC-AUC while maintaining balanced precision and recall across both classes. It was selected as the production model served by the API.

---

## Technology Stack

| Category | Tools |
|---|---|
| Language | Python |
| Machine Learning | Scikit-learn, Bernoulli Naive Bayes, Logistic Regression, TF-IDF |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Backend | FastAPI, Uvicorn, Pydantic |
| Deployment | Docker, Docker Hub, AWS EC2 |

---

## API Endpoints

### Health Check

**Request**
```http
GET /
```

**Response**
```json
{
    "status": "Sentiment Analysis API is running"
}
```

### Predict Sentiment

**Request**
```http
POST /predict
```

**Request Body**
```json
{
    "text": "I absolutely love this product! It exceeded all my expectations."
}
```

**Response**
```json
{
    "text": "I absolutely love this product! It exceeded all my expectations.",
    "sentiment": "positive",
    "confidence": 0.995
}
```

---

## Docker Deployment

### Build

```bash
docker build -t sentiment-api .
```

### Run

```bash
docker run -d \
  --name sentiment-api \
  -p 8000:8000 \
  sentiment-api
```

### Verify

```bash
docker ps
docker images
```

<img width="1568" height="94" alt="image" src="https://github.com/user-attachments/assets/87b53e7f-7407-4238-94bf-fd3e6376e6ec" />

<img width="1568" height="112" alt="image" src="https://github.com/user-attachments/assets/737ea310-7e20-469b-8c0a-826511bb2479" />

---

## AWS EC2 Deployment

Deployment workflow, end to end:

1. Train the model (`train.py`), save with Pickle
2. Build the Docker image
3. Push the image to Docker Hub
4. Launch an AWS EC2 Ubuntu instance
5. Install Docker on EC2
6. Pull the image from Docker Hub
7. Run the container
8. Open port `8000` in the EC2 security group
9. Access the API via the EC2 public IPv4 address

```bash
# On EC2, after Docker is installed:
docker pull devsuthar01/sentiment-api:latest
docker run -d -p 8000:8000 devsuthar01/sentiment-api:latest
```

Live at: `http://51.21.180.81:8000/docs`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/devsuthar01/sentiment-analysis.git
cd sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Train the model (generates model.pkl and vectorizer.pkl)
python train.py

# Run the API locally
uvicorn main:app --reload
```

Open Swagger docs at: `http://127.0.0.1:8000/docs`

---

## Project Structure

```text
sentiment-analysis/
│
├── main.py                 # FastAPI application
├── train.py                 # Training pipeline
├── text_utils.py             # Shared preprocessing (train + inference)
├── model.pkl                 # Trained Logistic Regression model
├── vectorizer.pkl            # Fitted TF-IDF vectorizer
├── Dockerfile
├── requirements.txt
├── README.md
│
└── images/
    ├── api-running.png
    ├── ec2-curl.png
    ├── swagger-home.png
    ├── swagger-prediction.png
    ├── docker-images.png
    ├── docker-ps.png
    ├── confusion-matrix-bnb.png
    ├── confusion-matrix-lr.png
    ├── roc-comparison.png
    └── wordcloud-negative.png
```

---

## Results Summary

- Trained and compared two classification models on 40,000 balanced tweets
- Logistic Regression outperformed Bernoulli Naive Bayes on both accuracy (76.60% vs. 75.75%) and ROC-AUC (0.840 vs. 0.829)
- Built and documented a REST API with FastAPI and Swagger/OpenAPI
- Containerized the application with Docker and published the image to Docker Hub
- Deployed to a live AWS EC2 instance, publicly accessible and verified via Swagger UI and direct `curl` requests

---

## Future Improvements

- Multi-class sentiment classification (Positive / Neutral / Negative)
- Transformer-based models (BERT, RoBERTa) for comparison against classical baselines
- Hyperparameter tuning (grid/random search)
- CI/CD pipeline via GitHub Actions (build → push → auto-deploy to EC2)
- MLflow for experiment tracking
- Model monitoring and logging in production

---

## Author

**Dev Suthar**
Machine Learning · NLP · FastAPI · Docker · AWS

- GitHub: [github.com/devsuthar01](https://github.com/devsuthar01)
- Docker Hub: [hub.docker.com/r/devsuthar01/sentiment-api](https://hub.docker.com/r/devsuthar01/sentiment-api)

---

## Acknowledgements

- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Scikit-learn · FastAPI · Docker · AWS

---

⭐ If you found this project useful, consider giving it a star on GitHub.
