from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

file_path = '../data/spam.csv'
model_path = '../data/model.sav'
encode_path = '../data/encoder.sav'

df = pd.read_csv(file_path, delimiter=',', encoding='latin1', usecols=['v1', 'v2'])
df.dropna(subset=['v1'], inplace=True)
df.fillna('', inplace=True)

X = df['v2']
y = df['v1']

le = LabelEncoder()
le.fit(y.unique())
y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', binary=True, lowercase=True, ngram_range=(1, 2))),
    ('k_best', SelectKBest(chi2, k=500)),
    ('model', MultinomialNB(alpha=0.1))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

joblib.dump(pipeline, model_path)
joblib.dump(le, encode_path)
