import os,re,sys
import glob

import pandas as pd



from TextFeatures.ReExtractTextFeatures import *

COMPANY_NAMES = ['Amazon', 'Apple', 'Delta', 'Google', 'Ibm', 'Microsoft', 'Pfizer', 'Spotify']


if __name__=='__main__':

    pathToStockInfoFiles = './Data/StockInfo/*.csv'

    root = './Data/Tweets/'

    for stockInfoFile in glob.glob(pathToStockInfoFiles):
        companyName = stockInfoFile.split('\\')[-1].split('.')[0]

        stocks = pd.read_csv(stockInfoFile)
        stocks['_Date'] = pd.to_datetime(stocks['Date']).dt.date




        for subDir, _, files in os.walk(root):
            if companyName.lower() in subDir.lower():
                print(companyName)

                Tweets = extractTextFeaturesFromTweet(subDir)

                globalSentiment = 0
                globalSubjectivity = 0
                GlobalSent, GlobalSubj = [], []

                Tweets['_Date'] = pd.to_datetime(Tweets['created_at']).dt.date

                """
                for i, tweet in Tweets.iterrows():

                    globalSentiment += tweet['emotion_score']
                    globalSubjectivity += tweet['subjectivity_score']

                    GlobalSent.append(globalSentiment)
                    GlobalSubj.append(globalSubjectivity)

                _GlobalSent = pd.Series(GlobalSent, index=Tweets.index)
                _GlobalSubj = pd.Series(GlobalSubj, index = Tweets.index)

                Tweets['global_sentiment'] = _GlobalSent
                Tweets['global_subjectivity'] = _GlobalSubj
                """

                for column in Tweets.columns:
                    if all(i in string.punctuation for i in column):
                        Tweets = Tweets.drop(column, axis=1)

                stocksAndTweets = pd.merge( Tweets, stocks, on='_Date')
                stocksAndTweets['Price'] = stocksAndTweets['Close']

                pathToCSVFile = './Data/StocksAndTweets/'+companyName + '.csv'
                stocksAndTweets.to_csv(pathToCSVFile, index=False, header=True)








