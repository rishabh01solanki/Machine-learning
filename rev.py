import os
# Set NLTK_DATA environment variable
os.environ['NLTK_DATA'] = '/Users/rishabhsolanki/Desktop/Machine learning/'

import re
import matplotlib.pyplot as plt
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import nltk
from nltk.corpus import movie_reviews as reviews



# Load the reviews and labels
X = [reviews.raw(fileid) for fileid in reviews.fileids()]
y = [reviews.categories(fileid)[0] for fileid in reviews.fileids()]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Define the tokenizer function
cachedStopwords = stopwords.words('english')
def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [w for w in words if w not in cachedStopwords]
    tokens = (list(map(lambda token:PorterStemmer().stem(token),words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
    return filtered_tokens

# Define the TF-IDF representation function
def represent(X_train,X_test):
    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    vectorized_train_documents = vectorizer.fit_transform(X_train)
    vectorized_test_documents = vectorizer.transform(X_test)
    return (vectorized_train_documents,vectorized_test_documents)

# Vectorize the reviews
train_docs, test_docs = represent(X_train,X_test)

# Convert labels from string to binary
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y_train)
train_labels = label_encoder.transform(y_train)
label_encoder.fit(y_test)
test_labels = label_encoder.transform(y_test)

# Prepare models
seed = 10
models = []
models.append(('Log Regression', LogisticRegression(random_state=seed)))
models.append(('Random Forest', RandomForestClassifier(random_state=seed)))
models.append(('Naive Bayes', MultinomialNB()))
models.append(('SVM', SVC(kernel='linear',random_state=seed,probability=True)))

classifiers = []
classifiers_names = []

plt.figure(figsize=(10, 8))

for name, model in models:
    clf = model
    clf.fit(train_docs, train_labels)
    classifiers.append(clf)
    classifiers_names.append(name)

    predictions = clf.predict(test_docs)
    predictions_prob = clf.predict_proba(test_docs)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(test_labels, predictions_prob[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='%s ROC (AUC = %0.2f)' % (name, roc_auc))
    
    # Confusion Matrix
    matrix = confusion_matrix(test_labels, predictions)
    print("\nConfusion matrix for ", name, ": ")
    print(matrix)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc.png')


def plot_confusion_matrices(classifiers, classifier_names, X_test, y_test):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))  # Adjust the figure size
    sns.set(font_scale=0.9)  # Adjust the font size
    axes = axes.flatten()
    for cls, name, ax in zip(classifiers, classifier_names, axes):
        cm = confusion_matrix(y_test, cls.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap='Blues', cbar=False)
        ax.set_title(f'Confusion Matrix: {name}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    plt.tight_layout(pad=5.0)
    plt.savefig('cfmtx.png')

plot_confusion_matrices(classifiers, classifiers_names, test_docs, test_labels)


