import os

# Set NLTK_DATA environment variable
os.environ['NLTK_DATA'] = '/Users/rishabhsolanki/Desktop/Machine learning/'

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import movie_reviews as reviews
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB



# Load the reviews and labels
X = [reviews.raw(fileid) for fileid in reviews.fileids()]
y = [reviews.categories(fileid)[0] for fileid in reviews.fileids()]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


cachedStopwords = stopwords.words('english')
def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [w for w in words if w not in cachedStopwords]
    tokens = (list(map(lambda token:PorterStemmer().stem(token),words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
    return filtered_tokens


def represent(X_train,X_test): # Define the TF-IDF representation function
    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    vectorized_train_documents = vectorizer.fit_transform(X_train)
    vectorized_test_documents = vectorizer.transform(X_test)
    return (vectorized_train_documents,vectorized_test_documents)


train_docs, test_docs = represent(X_train,X_test) # Vectorize the reviews


label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y_train)
train_labels = label_encoder.transform(y_train)
label_encoder.fit(y_test)
test_labels = label_encoder.transform(y_test)


def train_classifier(classifier, train_docs, train_labels):
    classifier.fit(train_docs, train_labels)
    return classifier

def make_predictions(classifier,train_docs,train_labels,test_docs):
    model = train_classifier(classifier,train_docs, train_labels)
    predictions = model.predict(test_docs)
    predictions_prob = model.predict_proba(test_docs)
    return(predictions,predictions_prob)

def evaluate(test_labels, predictions, predictions_prob, classifier):
    fpr, tpr, thresholds = roc_curve(test_labels, predictions_prob[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.8, label='%s ROC (AUC = %0.2f)' % (str(type(classifier)).split('.')[-1].replace('>','').replace("'",''), roc_auc))
    return roc_auc


seed = 10
models = [('Log Regression', LogisticRegression(random_state=seed)),
          ('Random Forest', RandomForestClassifier(random_state=seed)),
          ('Naive Bayes', MultinomialNB()),
          ('SVM', SVC(kernel='linear',random_state=seed,probability=True))]


results_dict = {'Classifier': [], 'ROC_AUC': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'Accuracy': []}


plt.figure(figsize=(10, 8))
results = []
names = []
results_dict = {'ROC_AUC': [], 'Classifier': [] }

for name, model in models:
    clf = model
    predictions, predictions_prob = make_predictions(clf, train_docs, train_labels, test_docs)
    roc = evaluate(test_labels, predictions, predictions_prob, clf)
    results.append(roc)
    names.append(name)
    results_roc = "Area under Curve ROC: " + "%s - %f" % (name, roc)
    results_dict['Classifier'].append(name)
    results_dict['ROC_AUC'].append(roc)
    print(results_roc)


df = pd.DataFrame(data=results_dict)
df.sort_values(by='ROC_AUC', ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)


df.style.background_gradient(cmap='Blues')
cm = sns.light_palette("green", as_cmap=True)
styled_table = df.style.background_gradient(cmap=cm)
styled_table

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for multiple classifiers')
plt.legend(loc="lower right")
plt.savefig('roc.png')
