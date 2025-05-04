# 📊 Sentiment Analysis on Flipkart Product Reviews

Analyze customer sentiments in Flipkart laptop reviews using machine learning and deep learning models.

## 📁 Dataset

**Source:** [Kaggle - Flipkart Laptop Reviews](https://www.kaggle.com/datasets/gitadityamaddali/flipkart-laptop-reviews)

**Columns:**
- `product_name`: Laptop name/specs
- `overall_rating`: Average product rating
- `no_ratings`: Number of ratings
- `no_reviews`: Number of reviews
- `rating`: User's rating (out of 5)
- `title`: Review summary
- `review`: Full review text

## 🛠 Features

- Preprocessing using `nltk` (stopwords, lemmatization, POS tagging)
- Feature extraction with `TF-IDF` and `Tokenizer`
- Traditional classification using `Logistic Regression`
- Deep learning with `LSTM` in TensorFlow/Keras
- Evaluation using accuracy and classification reports

## 📦 Libraries Used

- `polars`, `seaborn`, `matplotlib`
- `nltk`, `scikit-learn`, `tensorflow`
- `re`, `string`, `pandas` (implied using polars)

## 📈 Workflow

1. **Data Loading & Cleaning**: Remove noise, handle missing values
2. **Exploratory Data Analysis**: Rating distributions, word clouds, etc.
3. **Text Preprocessing**: Tokenization, lemmatization, stopword removal
4. **Modeling**:
   - Logistic Regression (TF-IDF features)
   - LSTM model (embedded padded sequences)
5. **Evaluation**: Compare classification metrics

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/flipkart-sentiment-analysis.git
cd flipkart-sentiment-analysis
pip install -r requirements.txt
jupyter notebook
