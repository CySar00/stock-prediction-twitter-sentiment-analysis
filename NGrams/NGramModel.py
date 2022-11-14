import math
import os,re,sys
import math
import collections
import ast

import nltk


import pandas as pd

from nltk.corpus import stopwords
nltk.download('stopwords')
_stopwords = set(stopwords.words('english'))

from collections import Counter

import json


MOST_COMMON = 10


"""
def seperateDataSetToTrainingAndTestSet(pathToCSVFile, trainingSetPercentage):

    # load the csv file
    dataFrame = pd.read_csv(pathToCSVFile, low_memory=False)

    trainingSetSize = math.ceil(trainingSetPercentage*len(dataFrame))
    testSetSize = len(dataFrame) - trainingSetSize

    trainingSet = dataFrame[:trainingSetSize]
    testSet = dataFrame[trainingSetSize+1:]

    return trainingSet, testSet
"""

def findMaxUnigrams(unigrams):

    Unigrams = []


    Words = []
    for uni in unigrams:
        _uni  = ast.literal_eval(uni)

        Words += _uni

    Dict = Counter(Words)
    common = Dict.most_common(MOST_COMMON)

    _common = [word[0] for word in common]

    return _common


def countUniGramFrequecy(unigrams):

    _Dict = []
    mostCommonWords = findMaxUnigrams(unigrams)
    UniDict = {}


    for uni in unigrams:
        _uni = uni.strip('[').strip(']').strip('"]')
        _uni = _uni.replace('(', '').replace('\'', '')

        words = _uni.split(',),')
        words = [word for word in words if word in mostCommonWords]

        for word in words:

            if word not in UniDict:
                UniDict[word] = 1

            else:
                UniDict[word] +=1

        for key in UniDict.keys():
            freq = round(UniDict[key]/len(UniDict), 4)

            UniDict[key] = freq

        _Dict.append(UniDict)

    return pd.DataFrame(_Dict)


def findMostCommonBigrams(bigrams):

    _Bigrams = []

    for bigram in bigrams:
        _bigram = ast.literal_eval(bigram)

        _Bigrams += _bigram

    Dict = Counter(_Bigrams)
    common = Dict.most_common(MOST_COMMON)

    _common = [word[0] for word in common]

    return _common



def countBiGramFrequency(bigrams):

    _Dict =[]
    mostCommonBigrams = findMostCommonBigrams(bigrams)

    BigramDict = {}
    for _bigrams in bigrams:
        _Bigrams = ast.literal_eval(_bigrams)

        for bi in _Bigrams:
            if bi in mostCommonBigrams:

                if bi not in BigramDict:
                    BigramDict[bi] = 1

                else:
                    BigramDict[bi] +=1


        for key in BigramDict.keys():

            freq = round(BigramDict[key]/len(BigramDict), 4)

            BigramDict[key] = freq

        _Dict.append(BigramDict)


    return pd.DataFrame(_Dict)

def findMostCommonTrigrams(trigrams):

    _Trigrams = []

    for trigram in trigrams:
        _trigram = ast.literal_eval(trigram)

        _Trigrams += _trigram

    Dict = Counter(_Trigrams)
    common = Dict.most_common(MOST_COMMON)

    _common = [word[0] for word in common]

    return _common



def countTrigramFrequency(trigrams):

    _Dict = []
    mostCommonTrigrams = findMostCommonBigrams(trigrams)

    TrigramDict = {}
    for _trigrams in trigrams:
        _Trigrams = ast.literal_eval(_trigrams)

        for tri in _Trigrams:
            if tri in mostCommonTrigrams:

                if tri not in TrigramDict:
                    TrigramDict[tri] = 1

                else:
                    TrigramDict[tri] += 1

        for key in TrigramDict.keys():
            freq = round(TrigramDict[key] / len(TrigramDict), 4)

            TrigramDict[key] = freq

        _Dict.append(TrigramDict)

    return pd.DataFrame(_Dict)


