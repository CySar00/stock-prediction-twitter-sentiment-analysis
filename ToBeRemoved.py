import os, re, sys


TO_BE_REMOVED = [ 'created_at', 'text', 'tweet_id', 'hashtags', 'mentioned_users', 'urls', 'author_screen_name',
                 'author_name', 'author_location', 'subjectivity', 'id', 'polarity_score', 'emotion', 'emotion_label',
                 'tweet', 'Date', '_Date', 'polarity', 'tokens', 'lemmas', 'stems', 'unigrams', 'bigrams', 'trigrams', 'parts_of_speech',
                 'sentiment', 'sentiment_class',  'subjectivity_class','parts_of_speech_features',
                  'global_sentiment_score', 'global_emotion_score', 'subjectivity_label', 'Price']


ADD = ['Date']

STOCKS = ['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Volume','Price', 'HighLow', 'HighLoad',  'Close', 'Label']

CLOSE = ['Change']