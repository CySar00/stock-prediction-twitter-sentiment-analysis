import os, re, sys
import unicodedata

import nltk

import num2word, num2words
import pandas as pd
import textblob as tb
import dateparser

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords
from dateparser.search import search_dates

import TextFeatures.Abbreviations
from TextFeatures.Abbreviations import ENGLISH_CONTRACTIONS, ENGLISH_CONTRACTIONS_NEW

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

_stopwords = set(stopwords.words('english'))
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()

sentimentIntensityAnalyzer = SentimentIntensityAnalyzer()


def removeRTFromTweet(tweet):
    if tweet.startswith('RT'):
        return tweet.replace('RT', '')

    return tweet


def removeUserMentionsFromTweet(tweet):
    return re.sub(r'@\S+', '', tweet)


def removeHashtagsFromTweet(tweet):
    return re.sub(r'[#]a-zA-Z0-9', r'', tweet)


def removeURLsFromTweet(tweet):
    return re.sub(r'http?://\S+|https?://\S+|www\.\S+', ' ', tweet)


def removeANSIISequencesFromTweet(tweet):
    tweet = tweet.replace(r'\\x[0-9a-fA-F]', '')

    ansi_regex = r'\x1b(' \
                 r'(\[\??\d+[hl])|' \
                 r'([=<>a-kzNM78])|' \
                 r'([\(\)][a-b0-2])|' \
                 r'(\[\d{0,2}[ma-dgkjqi])|' \
                 r'(\[\d+;\d+[hfy]?)|' \
                 r'(\[;?[hf])|' \
                 r'(#[3-68])|' \
                 r'([01356]n)|' \
                 r'(O[mlnp-z]?)|' \
                 r'(/Z)|' \
                 r'(\d+)|' \
                 r'(\[\?\d;\d0c)|' \
                 r'(\d;\dR))'

    ansi_escape = re.compile(ansi_regex, flags=re.IGNORECASE)

    tweet = re.sub(ansi_escape, ' ', tweet)

    return tweet


def removeXSequencesFromTweet(tweet):
    escape_char = re.compile(r'\\x[0123456789abcdef]{2,4}')
    tweet = re.sub(escape_char, '', tweet)

    return tweet


def removeLTSequencesFromTweet(tweet):
    tweet = tweet.replace('&lt;', ' ')
    tweet = tweet.replace('&amp;', ' ')
    tweet = tweet.replace('&gt', ' ')

    return tweet


def removeNewLinesFromTweet(tweet):
    tweet = re.sub(r'\\t', ' ', tweet)
    tweet = re.sub(r'\\v', ' ', tweet)
    tweet = re.sub(r'``', ' ', tweet)

    return re.sub(r'\\n', ' ', tweet)


def deEmojify(tweet):
    EMOJIS = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002500-\U00002BEF"  # chinese char
                        u"\U00002702-\U000027B0"
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        u"\U0001f926-\U0001f937"
                        u"\U00010000-\U0010ffff"
                        u"\u2640-\u2642"
                        u"\u2600-\u2B55"
                        u"\u200d"
                        u"\u23cf"
                        u"\u23e9"
                        u"\u231a"
                        u"\ufe0f"  # dingbats
                        u"\u3030"
                        "]+", re.UNICODE)

    return re.sub(EMOJIS, '', tweet)


def removeSpecialCharactersFromTweet(tweet):
    return re.sub('[^A-Za-z0-9]+', ' ', tweet)


def removeNumbersOfTweet(tweet):
    _tweet = ''.join([i for i in tweet if not i.isdigit()])
    return _tweet


def removeWhiteSpacesFromTweet(tweet):
    whiteSpaces = re.compile(r'\s+')

    return re.sub(whiteSpaces, ' ', tweet)


def removeAbbreviationsFromTweet(tweet):
    words = tweet.split()

    for word in words:
        if word in ENGLISH_CONTRACTIONS:
            tweet = tweet.replace(word, ENGLISH_CONTRACTIONS[word])

        if word in ENGLISH_CONTRACTIONS_NEW:
            tweet = tweet.replace(word, ENGLISH_CONTRACTIONS_NEW[word])

    return tweet


def tokenizeTheTweet(tweet):
    tokens = word_tokenize(tweet)

    specialCharacters = '\'\";:,./<>?[]{}|`~-_+=!@#$%^&*()//\'\'--!!??!!!???'

    _tokens = []
    for token in tokens:
        if token not in specialCharacters:
            _tokens.append(token)

    return _tokens


def stemTheTweet(tokens):
    stems = []
    for token in tokens:
        if token not in _stopwords:
            stems.append(_stemmer.stem(token))

    return stems


def lemmatizeTheTweet(tokens):
    lemmas = []
    for token in tokens:
        lemmas.append(_lemmatizer.lemmatize(token))

    return lemmas


def convertNumbersToText(tweet):
    words = tweet.split()
    _words = []
    for word in words:
        if word.isnumeric():
            _word = num2words.num2words(word)
            _words.append(_word)
        else:
            _words.append(word)

    _tweet = ' '.join([_w for _w in _words])
    return _tweet


def extractPartsOfSpeech(tokens):
    return nltk.pos_tag(tokens)


def extractFeaturesFromPartsOfSpeech(partsOfSpeech):
    _partsOfSpeech = dict(partsOfSpeech)

    partsOfSpeechCount = {}
    for key in _partsOfSpeech:

        if _partsOfSpeech[key] not in partsOfSpeechCount:
            partsOfSpeechCount[_partsOfSpeech[key]] = 1
        else:
            partsOfSpeechCount[_partsOfSpeech[key]] += 1

    for part in partsOfSpeechCount:
        partsOfSpeechCount[part] = round(partsOfSpeechCount[part] / len(_partsOfSpeech), 4)

    return partsOfSpeechCount


def extractNGramsFromTweet(ttokens):
    unigrams = nltk.ngrams(ttokens, 1)
    bigrams = nltk.ngrams(ttokens, 2)
    trigrams = nltk.ngrams(ttokens, 3)

    _unigrams, _bigrams, _trigrams = [], [], []

    for uni in unigrams:
        _unigrams.append(uni)

    for bi in bigrams:
        _bigrams.append(bi)

    for tri in trigrams:
        _trigrams.append(tri)

    return _unigrams, _bigrams, _trigrams


def extractSentimentOfTweet(tweet):
    tweet = unicodedata.normalize('NFKD', tweet)

    sentiment = tb.TextBlob(tweet).sentiment
    sentimentScore = sentiment[0]

    if sentimentScore > 0:
        _sentiment = 'positive'
        _class = 2

    elif sentimentScore < 0:
        _sentiment = 'negative'
        _class = 0

    else:
        _sentiment = 'neutral'
        _class = 1

    return _sentiment, sentimentScore, _class

def extractEmotionIntensity(tweet):


    _tweet = unicodedata.normalize('NFKD', tweet)

    emotion = sentimentIntensityAnalyzer.polarity_scores(_tweet)

    compound = emotion['compound']

    negative = emotion['neg']
    neutral = emotion['neu']
    positive = compound - neutral - negative


    return compound, neutral, negative, positive




def extractSubjectivityOfTweet(tweet):
    subjectivity_score = tb.TextBlob(tweet).subjectivity

    if subjectivity_score > 0.5:
        _subjectivity = 'subjective'
        _class = 1
    else:
        _subjectivity = 'objective'
        _class = 0

    return _subjectivity, subjectivity_score, _class


def removeStopwordsFromTweet(tweet):

    words = tweet.split(' ')

    tmp = [w for w in words if not w.lower() in _stopwords]

    tmp1 = []

    str = ''
    for word in tmp:

            if len(word) > 1:
                tmp1.append(word)
                str +=word + ' '


    return str





def preprocessTheTweet(tweet):

    tweet = tweet[2:-1]

    # remove 'RT' prefix from the beginning of the tweet
    _tweet = removeRTFromTweet(tweet)
    _tweet = removeNewLinesFromTweet(_tweet)
    _tweet = removeXSequencesFromTweet(_tweet)

    _tweet = removeLTSequencesFromTweet(_tweet)

    # remove the hashtags, urls, user mentions and emojis from the tweet
    _tweet = removeHashtagsFromTweet(_tweet)
    _tweet = removeUserMentionsFromTweet(_tweet)
    _tweet = removeURLsFromTweet(_tweet)
    _tweet = deEmojify(_tweet)

    _tweet = removeNewLinesFromTweet(_tweet)
    _tweet = removeXSequencesFromTweet(_tweet)
    _tweet = removeLTSequencesFromTweet(_tweet)
    _tweet = removeWhiteSpacesFromTweet(_tweet)

    _tweet = convertNumbersToText(_tweet)
    _tweet = removeSpecialCharactersFromTweet(_tweet)
    _tweet = removeNumbersOfTweet(_tweet)

    _lowerCaseTweet = _tweet.lower()
    _lowerCaseTweet = removeAbbreviationsFromTweet(_lowerCaseTweet)

    _stopwordsRemovedFromTweet = removeStopwordsFromTweet(_lowerCaseTweet)
    print(_stopwordsRemovedFromTweet)

    # tokenize the tweet
    _tokens = tokenizeTheTweet(_stopwordsRemovedFromTweet)

    # stem the tweet
    _stems = stemTheTweet(_tokens)

    # lemmatize the tweet
    _lemmas = lemmatizeTheTweet(_tokens)

    # extract parts of speech from the tweet
    partsOfSpeech = extractPartsOfSpeech(_tokens)

    featuresPartsOfSpeech = extractFeaturesFromPartsOfSpeech(partsOfSpeech)

    # extract n-grams from the tweet
    unigrams, bigrams, trigrams = extractNGramsFromTweet(_tokens)

    # extract the tweet's sentiment and subjectivity
    sentiment, sentimentScore, sentimentClass = extractSentimentOfTweet(_lowerCaseTweet)
    subjectivity, subjectivity_score, subj_class = extractSubjectivityOfTweet(_lowerCaseTweet)

    compound, negative, neutral, positive = extractEmotionIntensity(_lowerCaseTweet)


    Tweet = {
        'tweet': tweet,
        'number_of_words': len(tweet.split()),
        'tokens': _tokens,
        'number_of_tokens': len(_tokens),

        'stems': _stems,
        'number_of_stems': len(_stems),

        'lemmas': _lemmas,
        'number_of_lemmas': len(_lemmas),
        'parts_of_speech': partsOfSpeech,
        'number_of_parts_of_speech': len(partsOfSpeech),

        'unigrams': unigrams,
        'number_of_unigrams': len(unigrams),

        'bigrams': bigrams,
        'number_of_bigrams': len(bigrams),

        'trigrams': trigrams,
        'number_of_trigrams': len(trigrams),

        'emotion_score': sentimentScore,
        'emotion': sentiment,
        'emotion_label': sentimentClass,


        'compound_emotion' : compound,
        'negative_emotion': negative,
        'neutral_emotion': neutral,
        'positive_emotion': positive,

        'subjectivity_score': subjectivity_score,
        'subjectivity': subjectivity,
        'subjectivity_label': sentimentClass


    }

    Tweet.update(featuresPartsOfSpeech)

    _Tweet = pd.DataFrame.from_dict(Tweet, orient='index')
    _Tweet = _Tweet.transpose()

    return Tweet
