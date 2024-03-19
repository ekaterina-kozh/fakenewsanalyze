from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('fake_news.csv')

atrain, atest, btrain, btest = train_test_split(df['text'],  df.label, test_size=0.2,  random_state=7)

#Удаляем стоп-слова
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
train = tfidf.fit_transform(atrain)
test = tfidf.transform(atest)

#Обучаем модель
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(train, btrain)

ypred = pac.predict(test)
accuracy = accuracy_score(btest, ypred)
print(f'Оценка точности пассивно-агрессивного классификатора: {round(accuracy*100,2)}%')

#отчет о классификации и матрица ошибок
cf_matrix = confusion_matrix(btest,ypred)
ax = sns.heatmap(cf_matrix/ cf_matrix.sum(axis=1, keepdims=True), annot=True, cmap='Blues', fmt='.4f', square=True)

ax.set_title('Матрица ошибок\n\n');
ax.set_xlabel('\nПредсказанные метки')
ax.set_ylabel('Истинные метки')
