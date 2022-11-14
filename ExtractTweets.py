import os,re,sys
import Twitter
import string

from Twitter.ExtractTweets.ExtractTweetsFromAPI import *




if __name__=='__main__':
    keywords = [ 'spotify',  'google', 'ibm', 'amazon', 'delta','pfizer', 'apple', 'microsoft']

    keywords1 = ['spotify',  'google', 'ibm', 'amazon', 'delta','pfizer']

    startDate = datetime.datetime(2020, 1, 1, 0, 0, 0).replace(tzinfo=pytz.UTC)
    endDate = datetime.datetime(2021, 6, 25, 0, 0, 0).replace(tzinfo=pytz.UTC)

    for keyword in keywords:
        directoryName  = string.capwords(keyword)
        print(directoryName)

        pathToCSVFile = './Data/Tweets/' + directoryName +'/tweets_07_04_2020.csv'
        tweets = extractTweetsFromAPI( keyword,  pathToCSVFile, startDate, endDate)
