import os,re,sys
import tweepy

from tweepy import API, Cursor, OAuthHandler, Stream
from tweepy.streaming import StreamListener

API_KEYS = {
    'API_KEY' : 'VOLlCkMVjk2vtO5OzyZeMurnL',
    'API_SECRET': 'v7M7SVmluTbBX5A1PzvosTi3sLwY0cAadqfsizTfr68mesRPb7',

    'ACCESS_KEY': '99579271-dRir63cWbiiuwOIvXXIbxZTEQI9j3ov860mnVN3cI',
    'ACCESS_SECRET' : 'eM5uCrgcSAK1M9SHj7yrncBdVA3FSKD9I10PjvCF9Bf6d'
}


class TwitterAuthenticator():
    def authenticate_twitter_api(self):

        auth = OAuthHandler(API_KEYS['API_KEY'], API_KEYS['API_SECRET'])
        auth.set_access_token(API_KEYS['ACCESS_KEY'], API_KEYS['ACCESS_SECRET'])

        return auth


class TwitterListener(StreamListener):

    def __init__(self, fileToSaveTweets):
        self.fileToSaveTweets = fileToSaveTweets

    def on_data(self, raw_data):
        try:
            print(raw_data)
            with open(self.fileToSaveTweets, 'a') as tf:
                try:
                   # print(raw_data)
                    tf.write(raw_data)

                except tweepy.TweepError as ex:
                    if ex.reason =='Not authorized':
                        print('Unable to scrape tweet')

                    if ex.api_code==131:
                        print('Unable to scrape tweet')



            return True
        except BaseException as be:
            print('Error on_data %s' %str(be))


    def on_error(self, status):

        if status==420:
            return False

        print(status)



class TwitterStreamer():
    """
        Class for streaming and processing live tweets
    """
    def __init__(self):
        self.twitterAuthenticator = TwitterAuthenticator()


    def stream_tweets(self, fileToSaveTweets, keywords):
        listener = TwitterListener(fileToSaveTweets)

        auth = self.twitterAuthenticator.authenticate_twitter_api()
        stream = Stream(auth, listener)

        stream.filter(track=keywords)



class TwitterClient():
    def __init__(self):
        self.auth = TwitterAuthenticator().authenticate_twitter_api()
        self.twitter_client = API(self.auth, wait_on_rate_limit=True)

    def get_twitter_client(self):
        return self.twitter_client
