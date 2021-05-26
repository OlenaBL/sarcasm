from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

@app.route('/')

def home():
	return render_template('index.html')

@app.route('/',methods=['POST'])

def predict():
	data0 = pd.read_json('https://res.cloudinary.com/olena/raw/upload/v1621887358/Sarcasm_Headlines_Dataset.json', lines=True)

	#Second dataset
	data1 = pd.read_csv('https://raw.githubusercontent.com/OlenaBL/sarcasm/main/GEN-sarc-notsarc.csv')
	data1['class'] = data1['class'].replace({'notsarc':0, 'sarc':1})
	data1.rename(columns={'text': 'headline', 'class': 'is_sarcastic'}, inplace=True)
	
	#Third dataset
	data2 = pd.read_csv('https://raw.githubusercontent.com/OlenaBL/sarcasm/main/HYP-sarc-notsarc.csv')
	data2['class'] = data2['class'].replace({'notsarc':0, 'sarc':1})
	data2.rename(columns={'text': 'headline', 'class': 'is_sarcastic'}, inplace=True)
	
	#Appanding datasets
	data_merged = data0.append(data1)
	data = data_merged.append(data2)
	
	#Defining variables X and y
	#data['headline'] = data['headline'].apply(cleaning)
	
	#Removing stopwords, except useful for our research.
	stop = set(stopwords.words('english')) - set(['not', 'no', 'nor', "don't", 'very', 'down', 'most', 'over', 'such'])
	
	#Applying vectorizer
	from sklearn.feature_extraction.text import TfidfVectorizer
	vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stop)
	
	y = data['is_sarcastic']
	X = data['headline']
	X = vectorizer.fit_transform(data['headline'])
	
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
	
	from sklearn.svm import LinearSVC
	classifier = LinearSVC(dual=False, verbose=0)
	classifier.fit(X_train, y_train)
	classifier.score(X_train, y_train)
	
	if request.method == 'POST':
		message = request.form['headline']
		data = [message]
		vectorizer = vectorizer.transform(data).toarray()
		my_prediction = classifier.predict(vectorizer)
	return render_template('answer.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
