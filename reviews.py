import os

# Set NLTK_DATA environment variable
os.environ['NLTK_DATA'] = '/Users/rishabhsolanki/Desktop/Machine learning/'

# Then import nltk and load movie_reviews
import nltk
from nltk.corpus import movie_reviews as reviews
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


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

# Define function to train classifier
def train_classfier(train_docs, train_labels):
    classifier = linear_model.LogisticRegression(random_state=10)
    classifier.fit(train_docs, train_labels)
    return classifier

# Train and evaluate a logistic regression model
model = train_classfier(train_docs, train_labels)
predictions = model.predict(test_docs)
fpr, tpr, thresholds = roc_curve(test_labels, predictions)
roc_auc = auc(fpr, tpr)
print('ROC-AUC for Logistic Regression : %0.2f' % (roc_auc))

# Train and evaluate a regularized logistic regression model
classifier = linear_model.LogisticRegression(penalty='l2',random_state=10)
classifier.fit(train_docs, train_labels)
model = train_classfier(train_docs, train_labels)
predictions = model.predict(test_docs)
fpr, tpr, thresholds = roc_curve(test_labels, predictions)
roc_auc = auc(fpr, tpr)
print('ROC-AUC for Regularized Logistic Regression : %0.2f' % (roc_auc))

# Define functions for building model, making predictions and evaluation
def train_classifier(classifier, train_docs, train_labels):
    classifier.fit(train_docs, train_labels)
    return classifier

def make_predictions(classifier,train_docs,train_labels,test_docs):
    model = train_classifier(classifier,train_docs, train_labels)
    predictions = model.predict(test_docs)
    predictions_prob = model.predict_proba(test_docs)
    return(predictions,predictions_prob)

def evaluate(test_labels, predictions,predictions_prob):
    fpr, tpr, thresholds = roc_curve(test_labels, predictions_prob[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC (AUC = %0.2f)' % (roc_auc))
    plt.show()
    return(roc_auc)

# Prepare models
seed = 10
models = []
models.append(('Log Regression', LogisticRegression(random_state=seed)))
models.append(('Random Forest', RandomForestClassifier(random_state=seed)))
models.append(('Naive Bayes', MultinomialNB()))
models.append(('SVM', SVC(kernel='linear',random_state=seed,probability=True)))

# Define variables to hold results
results = []
names = []
results_dict = {'ROC_AUC': [], 'Classifier': [] }

# Evaluate each model in turn
for name, model in models:
    clf = model
    predictions,predictions_prob = make_predictions(clf, train_docs, train_labels, test_docs)
    roc = evaluate(test_labels, predictions,predictions_prob)
    results.append(roc)
    names.append(name)
    results_roc = "Area under Curve ROC: " + "%s - %f" % (name, roc)
    results_dict['Classifier'].append(name)
    results_dict['ROC_AUC'].append(roc)
    print(results_roc)

df = pd.DataFrame(data=results_dict)
print(df)
