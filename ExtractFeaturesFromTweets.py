import os,re,sys

import TextFeatures
from TextFeatures.Abbreviations import *
from TextFeatures.RePreprocessTheTweet import *
from TextFeatures.ReExtractTextFeatures import *
from TextFeatures.ASCII import *



if __name__=='__main__':
    extractTextFeaturesFromTweet('Data/Tweets/Spotify')
    print(1)