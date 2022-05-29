import pandas as pd
import nltk
from flask import request, jsonify, Flask, render_template
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def my_form_post():
    text = request.form['text']
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    score = ((sid.polarity_scores(str(text)))['compound'])

    if(score>0):
        label = 'This setence is positive'
    elif(score==0):
        label = 'This sentence is neutral'
    else:
        label = 'This sentence is negative'
    
    return render_template('index.html', variable=label)

if __name__ == "__main__":
    app.run(port='8088', threaded=False, debug=True)

