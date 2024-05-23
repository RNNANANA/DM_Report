import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import time
import re
from sklearn.ensemble import RandomForestClassifier

from file_io import *

emotion_list = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']

def convert_labels(labels):
    labels2 = []
    for idx, label in enumerate(labels):
        try: 
            label = str(label[0])
        except:
            label = str(label)
            pass
        temp = []
        num = len(emotion_list) - len(label)
        if (num == 0): 
            temp = [int(x) for x in label]
        else:
            temp = ''.join(['0']*num) + str(label)   
            temp = [int(x) for x in temp]
        labels2.append(temp)
    return labels2
    
train_set = read_list_from_jsonl_file('dataset/train_dataset.json')
val_set = read_list_from_jsonl_file('dataset/val_dataset.json')
test_set = read_list_from_jsonl_file('dataset/test_dataset.json')

train_set = train_set + test_set + val_set
print(len(train_set))

data_labels = [item['label_id'] for item in train_set]

df_labels = pd.DataFrame(data_labels, columns=['labels'])
print('df_labels: ', df_labels)

cleaned_data = []

lemma = WordNetLemmatizer()
stop_words = stopwords.words("english")
for item in train_set:
    text = item['text']
    
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
    text = re.sub("[^a-zA-Z0-9]"," ",text)
    text = nltk.word_tokenize(text.lower())
    text = [lemma.lemmatize(word) for word in text]
    text = [word for word in text if word not in stop_words]
    text = " ".join(text)
    
    cleaned_data.append(text)

for i in range(0, 5):
    print(cleaned_data[i], end="\n\n")

vectorizer = TfidfVectorizer()
BOW = vectorizer.fit_transform(cleaned_data)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
BOW = vectorizer.fit_transform(cleaned_data)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(BOW, np.asarray(df_labels), test_size=0.3, train_size=0.7, random_state=0)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

print(x_train.shape)
print(y_train.shape)

print(x_val.shape)
print(y_val.shape)

print(x_test.shape)
print(y_test.shape)

start_time = time.time()


model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(x_train, y_train)

end_time = time.time()
process_time = round(end_time - start_time, 2)
print("Fitting RandomForestClassifier took {} seconds".format(process_time))

predictions = model.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

print('y_test: ', y_test)
print('predictions: ', predictions)

y_test = convert_labels(y_test)
predictions = convert_labels(predictions)

f1_mi = f1_score(y_true=y_test, y_pred=predictions, average='micro')
re_mi = recall_score(y_true=y_test, y_pred=predictions, average='micro')
pre_mi = precision_score(y_true=y_test, y_pred=predictions, average='micro')
    
f1_mac = f1_score(y_true=y_test, y_pred=predictions, average='macro')
re_mac = recall_score(y_true=y_test, y_pred=predictions, average='macro')
pre_mac = precision_score(y_true=y_test, y_pred=predictions, average='macro')
    
result = {
    'f1_micro': f1_mi,
    'recall_micro': re_mi,
    'precision_micro': pre_mi,
    'f1_macro': f1_mac,
    'recall_macro': re_mac,
    'precision_macro': pre_mac
}

print(result)
