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
	data = pd.read_json('https://res.cloudinary.com/olena/raw/upload/v1621887358/Sarcasm_Headlines_Dataset.json', lines=True)

	from sklearn.feature_extraction.text import TfidfVectorizer#stop = set(stopwords.words('english')) - set(['not', 'no', 'nor', "don't", 'very', 'down', 'most', 'over', 'such'])
	vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')
	y = data['is_sarcastic']
	X = vectorizer.fit_transform(data['headline'])
	
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	
	from sklearn.naive_bayes import MultinomialNB
	classifier = MultinomialNB()
	classifier.fit(X_train, y_train)
	classifier.score(X_test,y_test)

	if request.method == 'POST':
		message = request.form['headline']
		data = [message]
		vectorizer = vectorizer.transform(data).toarray()
		my_prediction = classifier.predict(vectorizer)
	return render_template('answer.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
