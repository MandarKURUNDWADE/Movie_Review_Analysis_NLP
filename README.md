# IMDb Movie Review Sentiment Analysis

A machine learning model that classifies IMDb movie reviews as positive or negative with 88.6% accuracy.

## Problem Statement
The primary objective of this project is to build a machine learning classification model that
can predict the sentiment of IMDb movie reviews. The dataset contains a collection of movie
reviews, and each review is labeled as either positive or negative.

## Dataset Information
The IMDb dataset contains a large number of movie reviews, each labeled with either a positive or negative sentiment.
- **Text of the review:** The actual review provided by the user.
- **Sentiment label:** The sentiment of the review, either "positive" or "negative."

## Features
- Text preprocessing pipeline (cleaning, normalization, vectorization)
- Multiple model comparison (SVM, Logistic Regression, Random Forest, Naive Bayes)
- TF-IDF feature extraction (5000 most important words)
- Comprehensive evaluation metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebooks in order:
1. `1_data_exploration.ipynb`
2. `2_preprocessing.ipynb`
3. `3_model_training.ipynb`
4. `4_evaluation.ipynb`

Sample code for training the SVM model:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(train_text)

# Model training
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
```

## Results

**Best Model Performance (SVM):**
- Accuracy: 88.6%
- Precision: 0.89
- Recall: 0.88
- F1-Score: 0.89

**Confusion Matrix:**
|                | Predicted Negative | Predicted Positive |
|----------------|--------------------|--------------------|
| Actual Negative | 4,312              | 649                |
| Actual Positive | 504                | 4,535              |

## Project Structure

```
imdb-sentiment-analysis/
├── data/                   # Dataset files
├── notebooks/              # Jupyter notebooks
│   ├── 1_data_exploration.ipynb
│   ├── 2_preprocessing.ipynb
│   ├── 3_model_training.ipynb
│   └── 4_evaluation.ipynb
├── src/                    # Source code
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## License [MIT](https://choosealicense.com/licenses/mit/)

