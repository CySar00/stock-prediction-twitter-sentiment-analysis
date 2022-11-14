import os,re,sys
import pytz
import pandas as pd
import json
import GetOldTweets3 as got3

from Twitter.ExtractTweets.TweetPreprocessing.PreprocessTheTweet import *
from Twitter.ExtractUserProfile.ExtractUserTimelineFromAPI import *
from Twitter.TwitterAPI import *



def extractTweetsFromAPI(keywords, pathToCSVFile, startDate, endDate):

    twitterClient = TwitterClient()
    api = twitterClient.get_twitter_client()

    Tweets, Authors, IDs  = [], [],[]



    tweets = tweepy.Cursor(api.search, q=keywords, lang='en', since=startDate, until=endDate)

    Tweets = []
    for tweet in tweets.items():
        with open('./Data/Tweets/tweets_spotify.txt', 'a', encoding='utf-8') as _tweetFile:
            _tweet = tweet._json

            if _tweet['text'].startswith('RT'):
                isRT=1
            else:
                isRT=0

            if _tweet['text'].startswith('@'):
                isReplyTo=1
            else:
                isReplyTo = 0

            entities = _tweet['entities']

            hashtags, userMentions, URLs =  entities['hashtags'], entities['user_mentions'], entities['urls']
            _hashtags, _userMentions, _URLs = [], [], []

            for hashtag in hashtags:
                _hashtags.append(hashtag['text'])

            for userMention in userMentions:
                _userMentions.append({
                    'screen_name' : userMention['screen_name'],
                    'name' : userMention['name']
                })

            for URL in URLs:
                _URLs.append(URL['url'])


            # extract the 'features' from the tweet
            tweetFeatures = preprocessTheTweet(_tweet['text'])

            # extract information from the 'author's' profile or timeline
            authorName = _tweet['user']['screen_name']
            authorProfile = extractUserTimeLineFromAPI(authorName)




            Tweet = {
                'created_at':_tweet['created_at'],
                'id' : _tweet['id'],
                'text': _tweet['text'].encode('utf-8'),
                'is_retweet': isRT,
                'is_reply_to':isReplyTo,
                'hashtags': _hashtags,
                'mentioned_users':_userMentions,
                'urls' : _URLs,

                'number_of_words' : tweetFeatures['number_of_words'],
                'tokens': tweetFeatures['tokens'],
                'number_of_tokens': tweetFeatures['number_of_tokens'],
                'stems' :tweetFeatures['stems'],
                'number_of_stems':tweetFeatures['number_of_stems'],
                'parts_of_speech' :  tweetFeatures['parts_of_speech'],
                'parts_of_speech_features': tweetFeatures['parts_of_speech_features'],

                 'unigrams':tweetFeatures['unigrams'],
                 'number_of_unigrams': tweetFeatures['number_of_unigrams'],

                'bigrams': tweetFeatures['bigrams'],
                'number_of_bigrams': tweetFeatures['number_of_bigrams'],

                'trigrams': tweetFeatures['trigrams'],
                'number_of_trigrams': tweetFeatures['number_of_trigrams'],

                'polarity':tweetFeatures['sentiment'],
                'polarity_score':tweetFeatures['sentiment_score'],

                'subjectivity':tweetFeatures['subjectivity'],
                'subjectivity_score':tweetFeatures['subjectivity_score'],

                'author_screen_name' : authorProfile['screen_name'],
                'author_name': authorProfile['name'],

                'author_number_of_tweets': authorProfile['number_of_tweets'],
                'author_number_of_retweets':authorProfile['number_of_retweets'],
                'author_location' : authorProfile['location'],
                'author_number_of_hashtags_used':authorProfile['number_of_hashtags'],
                'author_number_of_conversations':authorProfile['number_of_conversations'],
                'author_number_of_favorite_tweets':authorProfile['number_of_favorite_tweets'],
                'author_number_of_lists':authorProfile['number_of_lists'],
                'author_number_of_followers' : authorProfile['number_of_followers'],
                'author_number_of_friends' : authorProfile['number_of_friends']

            }

            print(Tweet)

            Tweets.append(Tweet)
            _tweetFile.write(str(Tweet) + '\n')

    _Tweets = pd.DataFrame(Tweets)
    _Tweets.to_csv(pathToCSVFile, index=False, header=True)



if  __name__=='__main__':
    keywords = 'spotify'

    startDate = '2020-01-01'
    endDate = '2021-02-28'
    pathToCSVFile = './tweets_22_02_2021.csv'
    tweets = extractTweetsFromAPI( keywords, pathToCSVFile, startDate, endDate)
