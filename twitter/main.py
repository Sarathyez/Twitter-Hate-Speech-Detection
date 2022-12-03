import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import numpy

cv = CountVectorizer()
le = LabelEncoder()

app = Flask(__name__)



@app.route('/')
def home():
 return render_template('indexs.html',**locals())

@app.route('/predict',methods=['POST'])
def predict():
    
     # Read data
    print(request.form["text"])

    text = request.form["text"]

    
    le=LabelEncoder()
#     # Encode the feature(text)
    le.classes_ = numpy.load('classes.npy')
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer() 
    X = cv.fit_transform([text]).toarray() 
    model = pickle.load(open('model.pkl', 'rb'))
    cv = pickle.load(open('cv.pkl', 'rb'))


    if request.method == 'POST':

         txt = request.form['text']
         text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text) 
         text = re.sub(r'[[]]', ' ', text)   
         text = text.lower()
         t_o_b = cv.transform([txt]).toarray()
         print("goood",t_o_b.shape)
         language = model.predict(t_o_b) 
         corr_language = le.inverse_transform(language) 
    
         output = corr_language[0]
         # print (output)
         if output :
            tweetz= "Hate"
         else :
            tweetz= "Fair"

    #return render_template('index.html', prediction='Language is in {}'.format(tweetz))
    return render_template('submit.html', **locals())


if __name__ == "__main__":
     app.run(debug=True)