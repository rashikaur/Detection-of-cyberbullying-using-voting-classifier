from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import numpy as np
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
import string 
from nltk.stem import WordNetLemmatizer


app = Flask(__name__)

#Reading data

df= pd.read_csv("data.csv")

#Cleaning data

Tweet = []
Labels = []

for row in df["Tweet"]:
    #tokenize words
    words = word_tokenize(row)
    #remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    #remove stop words
    english_stops = set(stopwords.words('english'))
    characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    #Lematise words
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    Tweet.append(lemma_list)


df['label'] = df['Text Label'].map({'Non-Bullying': 0, 'Bullying': 1})
df['message']=df['Tweet']
df.drop(['Text Label','Tweet'],axis=1,inplace=True)
X = df['message']
y = df['label']

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Model creation
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = RandomForestClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
model4 = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
estimators.append(('sgd', model4))

warnings.simplefilter("ignore")
# create the ensemble model
classifier= VotingClassifier(estimators)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = classifier.predict(vect)
		
	return render_template('result.html',prediction = my_prediction,message=message)

if __name__ == '__main__':
	app.run(debug=True)