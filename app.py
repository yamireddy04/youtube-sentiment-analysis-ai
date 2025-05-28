from flask import Flask,render_template,redirect,request,url_for, send_file, session, Response, jsonify
import mysql.connector, joblib, random, string, base64, pickle
import pandas as pd
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
import re
import io
import base64
import matplotlib.pyplot as plt
import requests
import re
import emoji
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import httplib2
import requests

import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Preprocess the comment text
stop_words = set(stopwords.words('english'))

import matplotlib
matplotlib.use('Agg')  # This avoids using the Tkinter backend


app = Flask(__name__)
app.secret_key = 'youtube' 

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='youtube'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']

        if password == c_password:
            query = "SELECT email FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])

            if email not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)

                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']

        if email.lower() == "admin@gmail.com" and password == "admin":
            return redirect("/admin")
        
        query = "SELECT email FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email in email_data_list:
            query = "SELECT * FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password == password__data[0][3]:
                session["user_email"] = email
                session["user_id"] = password__data[0][0]
                session["user_name"] = password__data[0][1]

                return redirect("/home")
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')



@app.route('/home')
def home():
    return render_template('home.html')




### Youtube Comments sentiment analysis

## Retrieve youtube comments
def get_youtube_comments(video_id, api_key):
    comments = []
    url = f'https://www.googleapis.com/youtube/v3/commentThreads?key={api_key}&textFormat=plainText&part=snippet&videoId={video_id}&maxResults=100'

    while url:
        response = requests.get(url)
        data = response.json()

        for item in data.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                # 'author': comment['authorDisplayName'],
                'comment': comment['textDisplay'],
                # 'publishedAt': comment['publishedAt']
            })

        # Check if there's a next page
        url = data.get('nextPageToken')
        if url:
            url = f'https://www.googleapis.com/youtube/v3/commentThreads?key={api_key}&textFormat=plainText&part=snippet&videoId={video_id}&maxResults=100&pageToken={url}'
        else:
            url = None

    return comments


## Extract youtube video id
def extract_video_id(url):
    # Regular expression pattern to extract video ID from YouTube URL
    pattern = r'(?:https?://(?:www\.)?youtube\.com/(?:v/|watch\?v=|(?:e(?:mbed)?\/))([a-zA-Z0-9_-]+))'
    
    # Search for the pattern in the URL
    match = re.search(pattern, url)
    
    if match:
        return match.group(1)  # The video ID is in the first captured group
    else:
        raise ValueError("Invalid YouTube URL. Could not extract video ID.")
    

## Save comments into "Dataset/youtube_comments.csv" file
def save_comments_to_csv(comments, filename):
    df = pd.DataFrame(comments)
    df.to_csv(filename, index=False)


## Load LSTM model
model = load_model('Models/lstm_sentiment_model.h5')


## Tokenizer configuration
MAX_NUM_WORDS = 5000
MAX_SEQUENCE_LENGTH = 100
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)


## Pre-trained tokenizer setup (adjust based on your trained model)
def prepare_tokenizer():
    global tokenizer
    df = pd.read_csv(r'Dataset\GBcomments.csv', on_bad_lines='skip')
    df.dropna(subset=['comment_text'], inplace=True)
    tokenizer.fit_on_texts(df['comment_text'].values)
prepare_tokenizer()


## Text preprocessing function
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
def preprocess_text(comment):
    comment = comment.lower().strip()
    comment = re.sub(r'http\S+|www.\S+', '', comment)
    comment = emoji.replace_emoji(comment, replace='')
    comment = re.sub(r'[^a-zA-Z0-9\s]', '', comment)
    comment = " ".join([word for word in comment.split() if word not in stop_words])
    return comment


## Predict sentiment for a list of comments
def predict_sentiment(comments):
    processed_comments = [preprocess_text(comment) for comment in comments]
    sequences = tokenizer.texts_to_sequences(processed_comments)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict(data)
    sentiments = np.argmax(predictions, axis=1)
    sentiment_counts = np.bincount(sentiments, minlength=3)
    total = len(sentiments)
    # percentages = {
    #     'Positive': (sentiment_counts[1] / total) * 100,
    #     'Neutral': (sentiment_counts[2] / total) * 100,
    #     'Negative': (sentiment_counts[0] / total) * 100
    # }

    Positive = (sentiment_counts[1] / total) * 100,
    Neutral = (sentiment_counts[2] / total) * 100,
    Negative = (sentiment_counts[0] / total) * 100
    return Positive, Negative, Neutral


@app.route('/comment', methods = ["GET", "POST"])
def comment():
    if request.method == "POST":
        url = request.form['url']
        video_id = extract_video_id(url)
        api_key = 'AIzaSyAubERcFNfiXQ1oDL5HxQXSYihrEUTYc48' 

        comments = get_youtube_comments(video_id, api_key)
        save_comments_to_csv(comments, r'Dataset\youtube_comments.csv')

        df = pd.read_csv(r"Dataset\youtube_comments.csv")
        # df.drop(["author", "publishedAt"], axis=1, inplace=True)
        df.dropna(inplace=True)
        comments = df["comment"]

        Positive, Negative, Neutral = predict_sentiment(comments)
        print(Positive, Negative, Neutral)
        pie_chart = generate_pie_chart(Positive[0], Negative, Neutral[0])

        return render_template('comment.html', pie_chart = pie_chart)
    return render_template('comment.html')







### Youtube Video sentiment analysis

## Get text from youtube video
def get_transcript(video_id):
    try:
        # Extract transcript using YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ' '.join([entry['text'] for entry in transcript_list])
        return transcript
    except Exception as e:
        return f"Error retrieving transcript: {e}"
    
## Preprocess the text
import re
import nltk

# Download NLTK resources 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from youtube_transcript_api import YouTubeTranscriptApi



# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocessing_text(text):
    # 1. Text Segmentation (Tokenize sentences)
    sentences = nltk.sent_tokenize(text)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]

    # Flatten list of words
    words_flat = [word for sublist in words for word in sublist]
    
    # 2. Remove Unwanted Characters: Remove special symbols and non-ASCII characters
    words_cleaned = [re.sub(r'[^\x00-\x7F]+', '', word) for word in words_flat]

    # 3. Lowercasing
    words_cleaned = [word.lower() for word in words_cleaned]

    # 4. Remove Stop Words
    words_filtered = [word for word in words_cleaned if word not in stop_words]

    # 5. Lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words_filtered]

    # 6. Handling Numerical Data: Replace digits with <NUM> token
    words_final = [re.sub(r'\d+', '<NUM>', word) for word in lemmatized_words]

    # 7. Handling Punctuation: Remove punctuation from the cleaned words
    text_cleaned = ' '.join([re.sub(r'[^\w\s]', '', word) for word in words_final])

    return text_cleaned


## Analyze the sentiment for the preprocessed text
# Initialize VADER Sentiment Analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def sentiment(tokens):
    # Analyze sentiment of each word/token
    sentiment_scores = [sia.polarity_scores(word) for word in tokens]

    # Calculate the percentages of positive, negative, and neutral words
    positive_words = len([score for score in sentiment_scores if score['compound'] > 0.05])
    negative_words = len([score for score in sentiment_scores if score['compound'] < -0.05])
    neutral_words = len([score for score in sentiment_scores if -0.05 <= score['compound'] <= 0.05])

    total_words = len(tokens)
    positive_percentage = (positive_words / total_words) * 100
    negative_percentage = (negative_words / total_words) * 100
    neutral_percentage = (neutral_words / total_words) * 100

    return positive_percentage, negative_percentage, neutral_percentage



## Function to generate pie chart and return as base64
def generate_pie_chart(positive, negative, neutral):
    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive, negative, neutral]
    
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=['green', 'red', 'gray'], startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Save the plot to a BytesIO object and encode it as base64
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


@app.route('/video', methods=["GET", "POST"])
def video():
    if request.method == "POST":
        url = request.form['url']

        # Extract video ID from URL
        match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
        video_id = match.group(1) if match else None
        
        if video_id:
            transcript = get_transcript(video_id)
            if "Error" not in transcript:  # Check if transcript is successfully retrieved
                preprocessed_text = preprocessing_text(transcript)
                print("Preprocessed Text:", preprocessed_text)

                # Tokenize the preprocessed text and pass it to the sentiment function
                tokens = nltk.word_tokenize(preprocessed_text)
                positive_percentage, negative_percentage, neutral_percentage = sentiment(tokens)

                # Generate pie chart
                pie_chart = generate_pie_chart(positive_percentage, negative_percentage, neutral_percentage)
                return render_template('video.html', pie_chart = pie_chart)
            else:
                preprocessed_text = transcript  # Error message returned from get_transcript
                return render_template('video.html', message = preprocessed_text)
        else:
            preprocessed_text = "Invalid YouTube URL"
            return render_template('video.html', message = preprocessed_text)
        
    return render_template('video.html')






if __name__ == '__main__':
    app.run(debug = True)