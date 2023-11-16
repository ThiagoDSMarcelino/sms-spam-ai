from typing import List
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=',', encoding='latin1', usecols=['v1', 'v2'])
    df.dropna(subset=['v1'], inplace=True)
    df.fillna('', inplace=True)
    
    return df

def format_data(df: pd.DataFrame) -> List[pd.Series]:
    X = df['v2']
    y = df['v1']
    
    vectorizer = CountVectorizer(stop_words='english', lowercase=True, binary=True)
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    k_best = SelectKBest(chi2, k=2)
    X_train = k_best.fit_transform(X_train, y_train)
    X_test = k_best.transform(X_test)

    return [X_train, X_test, y_train, y_test]

def gen_model(data: List[pd.Series]) -> MultinomialNB:
    X_train, X_test, y_train, y_test = data
    
    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    
    return model

def main():
    file_path = 'src/data/spam.csv'
    model_path = 'src/data/model.sav'

    df = load_df(file_path)
    
    data = format_data(df)

    model = gen_model(data)
    
    joblib.dump(model, model_path)

if __name__ == "__main__":
    main()
