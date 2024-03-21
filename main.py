from sklearn.model_selection import train_test_split

import preprocess

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def word_cloud(word_cloud_data):
    plt.figure(figsize=(20, 20))
    wc = WordCloud(max_words=1000, width=1600, height=800,
                   collocations=False).generate(" ".join(word_cloud_data))
    plt.imshow(wc)

    plt.axis('off')

    plt.show()  # <-- This line displays the figure


def model_evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative', 'Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)


data = preprocess.load_processed_data("twitter/processed/cleanTwitterData.csv")

X = data['text']
y = data['target']

# Separating the 95% data for training data and 5% for testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=26105111)

vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)

vectoriser.fit(X_train)
num_features = len(vectoriser.vocabulary_)
print("Number of features: %d" % num_features)
X_train = vectoriser.transform(X_train)
X_test = vectoriser.transform(X_test)

# Instantiate the Logistic Regression model
model = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)

# Train the model on your training data
model.fit(X_train, y_train)
model_evaluate(model)



