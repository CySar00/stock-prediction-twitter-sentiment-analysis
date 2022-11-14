import os,re,sys
import glob

import pandas as pd
import nltk
import glob


from NGrams.NGramModel import *



if __name__=='__main__':

    #pathToSpotifyDataFrame = './Data/StocksAndTweets/Spotify.csv'

    pathToCSVFiles = './Data/StocksAndTweets/*.csv'

    for csvFile in glob.glob(pathToCSVFiles):
        companyName = csvFile.split('\\')[-1].split('.')[0]
        print(companyName)

        _csvFile = './Data/NGrams/'+ companyName + '.csv'


        dataFrame = pd.read_csv(csvFile, low_memory=False)
        print(dataFrame['tokens'])
        unigrams, bigrams, trigrams = dataFrame['unigrams'], dataFrame['bigrams'], dataFrame['trigrams']

        UnigramFrequencyCount = countUniGramFrequecy(unigrams)
        BigramFrequencyCount = countBiGramFrequency(bigrams)
        TrigramFrequencyCount = countTrigramFrequency(trigrams)

        # concatenate
        dataFrame = pd.concat(
            [dataFrame.set_index(dataFrame.index), UnigramFrequencyCount.set_index(dataFrame.index),
             BigramFrequencyCount.set_index(dataFrame.index), TrigramFrequencyCount.set_index(dataFrame.index)],
            axis=1)

        dataFrame = dataFrame.fillna(0)
        dataFrame.to_csv(_csvFile, index=False, header=True)
        print(dataFrame)


        #_DataFrames.to_csv(_csvFile, index=False, header=True)








