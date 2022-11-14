import os,re,sys
import tweepy

import pandas as pd
import datetime, dateutil

from dateutil.parser import parse


import Twitter
from Twitter.TwitterAPI import *
from Twitter.APICredentials import *


def extractUserTimeLineFromAPI(screenName):

    twitterClient = TwitterClient()
    api = twitterClient.get_twitter_client()


    authorProfile = api.get_user(screenName)

    _authorProfile = authorProfile._json


    authorName = _authorProfile['name']
    authorLocation = _authorProfile['location']

    authorNumberOfFollowers  = _authorProfile['followers_count']
    authorNumberOfFriends = _authorProfile['friends_count']
    authorNumberOfTweets = _authorProfile['statuses_count']

    authorNumberOfFavorites = _authorProfile['favourites_count']
    authorLists = _authorProfile['listed_count']


    tweets = api.user_timeline(screen_name=screenName, count=authorNumberOfTweets)

    numberOfRTs, numberOfConversations, totalNumberOfHashtags = 0, 0, 0

    Tweets = []
    for tweet in tweets:
        _tweet = tweet._json
        print(_tweet)

        hashtags = _tweet['entities']['hashtags']
        totalNumberOfHashtags+=len(hashtags)

        if _tweet['text'].startswith('@'):
            numberOfConversations+=1

        if _tweet['text'].startswith('RT'):
            numberOfRTs+=1

        Tweet = {
            'created_at' : parse(_tweet['created_at']),
            'tweet_id' : _tweet['id']
        }

        Tweets.append(Tweet)

    _Tweets = pd.DataFrame(Tweets)
    try:
        _Tweets['date'] = pd.to_datetime(_Tweets['created_at'])
        avgTweetsPerDay = _Tweets.groupby('date').count().mean()['tweet_id']

    except KeyError as ke:
        _Tweets['date'] = pd.to_datetime('now')
        avgTweetsPerDay=1
        print(ke)





    authorProfile = {
        'screen_name':screenName,
        'name':authorName,
        'location':authorLocation,
        'number_of_tweets':authorNumberOfTweets,
        'number_of_retweets':numberOfRTs,
        'number_of_conversations':numberOfConversations,
        'number_of_hashtags':totalNumberOfHashtags,
        'number_of_followers':authorNumberOfFollowers,
        'number_of_friends':authorNumberOfFriends,
        'number_of_favorite_tweets': authorNumberOfFavorites,
        'number_of_lists':authorLists,
        'tweeting_frequency': avgTweetsPerDay
    }



    return authorProfile


















if __name__=='__main__':


    screenName = '@wesrap'

    authorInfo = extractUserTimeLineFromAPI(screenName)
    print(authorInfo)