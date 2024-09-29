
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK stopwords
nltk.download("stopwords")

# Reading the dataset (use raw string to avoid escape sequences)
message = pd.read_csv(r"d:\spam\sms_spam_collection.csv", sep=',', names=['label', 'message'])

# Data cleaning and preprocessing
ps = PorterStemmer()
corpus = []
for i in range(0, len(message)):
    # Keep only letters and replace everything else with spaces
    review = re.sub('[^a-zA-Z]', ' ', message['message'][i])
    review = review.lower()
    review = review.split()
    
# Stemming and removing stopwords
    review = [ps.stem(word) for word in review if word not in stopwords.words("english")]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

# Label encoding: converting 'ham'/'spam' to binary values
y = (message['label'] == 'spam').astype(int)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training the model using Naive Bayes classifier
spam_detect_model = MultinomialNB().fit(X_train, y_train)

# Predicting on the test set
y_pred = spam_detect_model.predict(X_test)

# Calculating accuracy
print("Accuracy: ", accuracy_score(y_test, y_pred))



