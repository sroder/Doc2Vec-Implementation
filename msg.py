#import pandas
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, KFold
from sklearn.externals import joblib
import numpy as np
#from utils import preprocess

xls = pd.ExcelFile('laufzeit_vertarg.xlsx')
df = pd.read_excel(xls,'Sheet3')
# print(df['Skript'].unique())

#df = preprocess(df)

X_train, X_test, y_train, y_test  = train_test_split(df['Beschreibung'],df['Label'], test_size=0.33, random_state=42)

model = LogisticRegression()
# model = LinearSVC()

vectorizer = TfidfVectorizer(min_df=5,
                             max_df=0.8,
                             sublinear_tf=True,
                             use_idf=True, ngram_range=(1, 2))
# train_vectors = vectorizer.fit_transform(X_train)
# test_vectors = vectorizer.transform(X_test)
kf = KFold(n_splits=10, random_state=43, shuffle=True)
print("Training Data")
# model.fit(train_vectors, y_train)
# prediction = model.predict(test_vectors)
# print(accuracy_score(y_test, prediction))
# print(precision_recall_fscore_support(y_test, prediction, average='macro'))
accurs = []
precision = []
recall = []
f1 = []
for train_index, test_index in kf.split(df):
    X_train, X_test = df.iloc[train_index]['Beschreibung'], df.iloc[test_index]['Beschreibung']
    y_train, y_test = df.iloc[train_index]['Label'], df.iloc[test_index]['Label']

    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)
    print("Training Data")
    model.fit(train_vectors, y_train)
    prediction = model.predict(test_vectors)
    accur = accuracy_score(y_test, prediction)
    print("Score ",accur)
    accurs.append(accur)
    cm1,cm2,cm3,_ = precision_recall_fscore_support(y_test, prediction, average='macro')
    precision.append(cm1)
    recall.append(cm2)
    f1.append(cm3)

joblib.dump(model, 'model.pkl') 
print(np.mean(accurs))
print(np.mean(precision))
print(np.mean(recall))
#print(f1)


    
