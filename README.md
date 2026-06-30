# Amazon Review Sentiment Analyzer

## Project Overview

This project is a Machine Learning-based Sentiment Analysis application that classifies Amazon product reviews as **Positive** or **Negative**. The model is trained on the Amazon Reviews dataset and deployed using Streamlit, allowing users to enter a review and instantly receive a sentiment prediction.

The project uses:

* TF-IDF Vectorization
* N-Gram Features (Unigrams, Bigrams, and Trigrams)
* Linear Support Vector Classifier (LinearSVC)
* Streamlit for Web Deployment
* Pickle (`model.pkl`) for model persistence

---

## Features

* Classifies reviews into Positive or Negative sentiment
* Supports real-time prediction through a Streamlit web interface
* Uses saved model (`model.pkl`) to avoid retraining on every startup
* Handles phrases such as:

  * "good"
  * "very good"
  * "not good"
  * "not very good"
* Lightweight and easy to deploy

---

## Dataset

Dataset: Amazon Reviews Dataset
Link: https://www.kaggle.com/datasets/deniyulian/sentiment-analysis

Columns used:

* `Text` → Review text
* `Score` → Rating score

Sentiment Mapping:

| Score | Sentiment         |
| ----- | ----------------- |
| 1, 2  | Negative          |
| 3     | Neutral (Removed) |
| 4, 5  | Positive          |

---

## Model Architecture

### Text Preprocessing

* Convert text to lowercase
* Remove URLs
* Remove HTML tags
* Remove special characters
* Remove extra spaces

### Feature Extraction

TF-IDF Vectorizer with:

* N-Gram Range: (1,3)
* Unigrams
* Bigrams
* Trigrams

Examples:

* good
* very good
* not good
* not very good

### Classification Model

LogisticRegression

I just selected this model , however , would later try train on svm , as it perform better for such tasks

---

## Project Structure

```text
senti-analysis/
│
├── app.py
├── train_model.py
├── test_model.py
├── requirements.txt
├── model.pkl
├── README.md

```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/senti-analysis.git
```

Move into the project directory:

```bash
cd senti-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

If you want to retrain the model:

```bash
python train_model.py
```

This will generate:

```text
model.pkl
```

which stores the trained model for future predictions.

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the browser:

```text
http://localhost:8501
```

Enter a review and click **Analyze** to view the sentiment prediction.

---

## Example Predictions

Input:

```text
This product is amazing and works perfectly.
```

Output:

```text
Positive Review
```

Input:

```text
This product is not good and stopped working after one day.
```

Output:

```text
Negative Review
```

---

## Deployment

This application can be deployed using:

* Streamlit Community Cloud
* Render
* Railway
* Hugging Face Spaces

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Pickle

---

## Future Improvements

* Fine-tune DistilBERT or BERT for higher accuracy
* Multi-class sentiment classification (Positive, Neutral, Negative)
* Aspect-Based Sentiment Analysis
* Sentiment confidence visualization
* User review history and analytics dashboard
* SVM instead of Logistic Regression

---

## Author

Developed as a Machine Learning and Natural Language Processing project for sentiment classification of Amazon product reviews.
