import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

file_path = 'src/data/spam.csv'
model_path = 'src/data/model.sav'

df = pd.read_csv(file_path, delimiter=',', encoding='latin1', usecols=['v1', 'v2'])
df.dropna(subset=['v1'], inplace=True)
df.fillna('', inplace=True)

X = df['v2']
y = df['v1']

vectorizer = CountVectorizer(stop_words='english', lowercase=False, binary=True)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k_best = SelectKBest(chi2, k=2)
X_train_best = k_best.fit_transform(X_train, y_train)
X_test_best = k_best.transform(X_test)

model = MultinomialNB()
model.fit(X_train_best, y_train)

y_pred = model.predict(X_test_best)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

joblib.dump(model, model_path)