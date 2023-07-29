# Sentiment Analysis on Movie Reviews using NLP and Machine Learning

In this project, we employ various machine learning algorithms to perform sentiment analysis on a corpus of movie reviews. Our goal is to classify movie reviews into "positive" or "negative" sentiments based on the textual content of the reviews.

## Table of Contents
1. [Data Ingestion](#data-ingestion)
2. [Preprocessing and Feature Extraction](#preprocessing-and-feature-extraction)
3. [Model Training and Evaluation](#model-training-and-evaluation)
4. [Results and Conclusion](#results-and-conclusion)

## Data Ingestion <a name="data-ingestion"></a>
The data for this project is sourced from the `movie_reviews` corpus of the NLTK library. This corpus comprises movie reviews labelled as either 'positive' or 'negative'.

## Preprocessing and Feature Extraction <a name="preprocessing-and-feature-extraction"></a>
To prepare the data for our machine learning models, we perform the following preprocessing steps:
- **Tokenization:** We convert the raw text of each review into a list of individual words or "tokens".
- **Stopword Removal:** We remove common words like "the", "and", "in", etc., that don't carry much semantic meaning.
- **Stemming:** We reduce each word to its base or "stem" form, which helps us generalize across different forms of the same word.
- **Vectorization:** We convert the processed text into numerical feature vectors using the TF-IDF method. 

## Model Training and Evaluation <a name="model-training-and-evaluation"></a>
We train and evaluate several machine learning models including Logistic Regression, Random Forest, Naive Bayes, and Support Vector Machines (SVM). 

For each model, we perform the following steps:
- **Model Training:** We train the model on our preprocessed feature vectors.
- **Prediction:** We use the trained model to make sentiment predictions on the test data.
- **Evaluation:** We evaluate the model's performance by calculating the Area Under the ROC Curve (AUC-ROC).

We visualize the ROC curves of the different models together for comparison.

## Results and Conclusion <a name="results-and-conclusion"></a>
The results of the model performance are presented in terms of ROC_AUC for each classifier. We find that [insert the model which worked best] outperforms the other models. 

Through this project, we successfully automate the process of sentiment analysis on movie reviews, which has numerous practical applications in the real world, from customer feedback interpretation to product recommendation.

We hope to further enhance the model by experimenting with other machine learning techniques and fine-tuning the hyperparameters of our current models.

