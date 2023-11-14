import string
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split

def get_stop_word(path: str) -> List[str]:
    stop_words = []

    with open(path, 'r', encoding='utf-8') as file:
        stop_words = [line.strip() for line in file]
        
    return stop_words

def remove_stop_words(sentence: str) -> str:
    stop_words = get_stop_word('stop_words.txt')
    
    words = sentence.split()
    words = [''.join(c for c in word if c not in string.punctuation) for word in words]
    filtered_sentence = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_sentence)

def get_all_words(df: pd.DataFrame) -> set[str]:
    keys = df.keys()
    
    all_words = set(df[keys[0]].dropna().str.split().dropna().sum())
    
    for key in keys[1:]:
        all_words |= set(df[key].dropna().str.split().dropna().sum())
        
    return all_words
    
def gen_df_based_on_words(df: pd.DataFrame) -> pd.DataFrame:
    words = get_all_words(df)
    
    new_df = pd.DataFrame()
    
    new_df = pd.concat([df.apply(lambda row: any(map(lambda sentence: isinstance(sentence, str) and word in sentence, row)), axis=1) for word in words], axis=1)
    
    new_df = new_df.replace({True: 1, False: 0})
    
    return new_df

file_path = "spam.csv"

df = pd.read_csv(file_path, delimiter=',', encoding="latin1")
df.fillna('', inplace=True)

label_header = "v1"

df["v2"] = df["v2"].apply(remove_stop_words)

X = df.drop(label_header, axis=1)
X = gen_df_based_on_words(X)

y = df[label_header]

X_train, X_test, y_train, y_test = \
  train_test_split(X, y, test_size=0.2, random_state=21)
