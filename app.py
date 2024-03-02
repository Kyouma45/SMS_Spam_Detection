import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
nltk.download('stopwords')

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

ip=st.text_area('Enter the message')

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text.clear()        
    for i in y:
        if i not in stopwords.words('english') and i not in string.punctuation:
            text.append(i)
            
    y.clear()
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

if st.button('Predict'):
	#preprocess
	ip_transformed=transform_text(ip)
	#vectorize
	vector_ip=tfidf.transform([ip_transformed]).toarray()
	#predict
	result=model.predict(vector_ip)

	if result==1:
		st.header('Spam!!!')
	else:
		st.header('Not Spam')