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

- Preprocessing using `nltk` (stopwords, lemmatization with POS tagging)
- Feature extraction with `TF-IDF` and `Tokenizer`
- Traditional classification using `Logistic Regression`
- Deep learning with `LSTM` in TensorFlow/Keras
- Evaluation using accuracy and classification reports

## 📦 Libraries Used

- `polars`, `seaborn`, `matplotlib`
- `nltk`, `scikit-learn`, `tensorflow`
- `re`, `string`, `pandas` (implied using polars)

## 📈 Workflow

1. **Data Loading & Cleaning**:
   - Checked for missing values.
   - Adjusted colunms values by remove ',' from **no_rating** column.
   - Based on overall_reviews and no_rating values, class imbalance was approximately removed, by removing the values outside of range.
     - Visualized the rating distribution using a bar chart for better visualization of data.
     - By implementing **Inter-Quartile Range** the values were located and adjusted.
     - For eg: For user rating 4: Only the 4th quartile value was taken as it had most number of reviews and overall ratings.
3. **Added Sentiments to reviews based on Rating**:
   - Converted rating to sentimental value. (1-2): **negative**, 3: **neutral** and (4-5) as **positive**. (Values are inclusive)
5. **Text Preprocessing**:
   - Removed emojis and stopwords and converted all the texts to lowercase for better vectorization and embedding.
   - Removed punctuations.
   - Performed **Tokenization** on the texts and then performed **Lemamtization with POS Tagging** to convert the words to their root form.
   - Splitted the dataset to Train, Test sets to prevent **Data Leakage**.
   - For Linear model:
     - **TF-IDF Vectorization** was performed.
     - **Label Encoding** was used for target variable.
   - For LSTM (DL) model the texts were **Tokenized**, then converted to **Sequences** and then **Padding** was added.
     - **One Hot Encoding** was used for target variable.
7. **Model Building**:
   - Logistic Regression with _TF-IDF_ features.
   - LSTM model with _embedded padded sequences_ .
8. **Evaluation**: Compared accuracy and classification reports.

## 🚀 How to Run

```bash
git clone https://github.com/Avik816/flipkart-sentiment-analysis.git
cd flipkart-sentiment-analysis
pip install -r requirements.txt
jupyter notebook
