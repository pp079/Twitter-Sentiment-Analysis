import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tweepy

# Twitter API credentials
consumer_key = '##################################'
consumer_secret = '################################'
access_token = '#####################################'
access_secret = '###################################'

# authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

# set search parameters
search_term = 'Tesla'
num_tweets = 100
lang = 'en'

# get tweets
tweets = tweepy.Cursor(api.search_tweets,
                       q=search_term,
                       lang=lang).items(num_tweets)

# analyze sentiment of tweets
vader = SentimentIntensityAnalyzer()
sentiment_scores = []
for tweet in tweets:
    scores = vader.polarity_scores(tweet.text)
    sentiment = scores['compound']
    sentiment_scores.append(sentiment)
  
# calculate percentages of positive, neutral, and negative sentiment
positive = [score for score in sentiment_scores if score > 0]
neutral = [score for score in sentiment_scores if score == 0]
negative = [score for score in sentiment_scores if score < 0]

positive_sentiment = len(positive) / len(sentiment_scores)
neutral_sentiment = len(neutral) / len(sentiment_scores)
negative_sentiment = len(negative) / len(sentiment_scores)

# plot pie chart of sentiment distribution
labels = ['Positive', 'Neutral', 'Negative']
sizes = [positive_sentiment, neutral_sentiment, negative_sentiment]
colors = ['#00ff00', '#808080', '#ff0000']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

plt.title(f"Sentiment Analysis of {search_term} Tweets")
plt.show()
