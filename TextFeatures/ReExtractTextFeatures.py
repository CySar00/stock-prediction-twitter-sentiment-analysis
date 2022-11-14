import os,re,sys
import ast
import glob
import csv
import codecs
import string

import pandas as pd
import pandas.errors

from TextFeatures.RePreprocessTheTweet import *
from TextFeatures.Abbreviations import *


def extractTextFeaturesFromTweet(pathToCSVFiles):

    csvFiles = glob.glob(pathToCSVFiles+'/*.csv')
    Tweets = []

    globalSentiment = 0
    for csvFile in csvFiles:

        try:
            with open(csvFile, 'rb+') as f:
                tweets = pd.read_csv(f)

                for i, tweet in tweets.iterrows():


                    tweet= tweet.drop(['tokens', 'number_of_tokens', 'stems', 'number_of_stems', 'parts_of_speech', 'unigrams', 'bigrams', 'trigrams'])

                    tweet['number_of_hashtags'] = len(tweet['hashtags'])
                    tweet['number_of_mentioned_users'] = len(tweet['mentioned_users'])
                    tweet['number_of_urls'] = len(tweet['urls'])


                    _tweet = tweet.to_dict()

                    tweetFeatures = preprocessTheTweet(tweet.text)

                    mergedFeatures = _tweet
                    mergedFeatures.update(tweetFeatures)
                    Tweets.append(mergedFeatures)


        except pandas.errors.EmptyDataError as pdee:
            continue


    _Tweets = pd.DataFrame(Tweets)
    _Tweets = _Tweets.fillna(0)

    print(_Tweets.columns)
    return _Tweets