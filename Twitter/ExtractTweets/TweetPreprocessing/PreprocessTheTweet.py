import os,re,sys
import nltk, analysis
import textblob as tb


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer




nltk.download('stopwords')
_stopwords = set(stopwords.words('english'))
_stemmer = PorterStemmer()
_lemmatizer  = WordNetLemmatizer()

def lowercaseTheTweet(tweet):
    return  tweet.lower()


def removeUserMentionsFromTweet(tweet):
    return re.sub(r'@\S+', '', tweet)

def removeURLsFromTweet(tweet):

    return re.sub(r'http?\://|https?\://|www\S+', '', tweet)


def removeHashtagsFromTweet(tweet):
    return  re.sub(r'#\w+', '', tweet)


def removeRTFromTweet(tweet):

    if tweet.startswith('rt'):
        return tweet[2:]
    else:
        return  tweet


def removeSpecialCharatctersFromTweet(tweet):

    tweet = re.sub('[^A-Za-z0-9]', ' ', tweet)
    tweet = re.sub('\s\s+' , '', tweet)


    tweet = ''.join([word for word in tweet if ord(word)<128])

    return tweet

def removeEmojisFromTweet(tweet):
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


def replaceContractedWords(tweet):
    _tweet = tweet.split()

    for word in _tweet:
        if word in ENGLISH_CONTRACTIONS:
            tweet = tweet.replace(word, ENGLISH_CONTRACTIONS[word])

        if word in ABBREVIATIONS_MONTHS:
            tweet = tweet.replace(word, ABBREVIATIONS_MONTHS[word])

    return tweet


def extractSentimentOfTweet(tweet):

    sentiment = tb.TextBlob(tweet).sentiment
    sentimentScore = sentiment[0]

    if sentimentScore>0:
        _sentiment = 'positive'

    elif sentimentScore<0:
        _sentiment = 'negative'

    else:
        _sentiment  = 'neutral'

    return sentimentScore, _sentiment

def extractSubjectivityOfTweet(tweet):
    subjectivity = tb.TextBlob(tweet).subjectivity

    if subjectivity>0.5:
        _subjectivity = 'subjective'
    else:
        _subjectivity = 'objective'

    return subjectivity, _subjectivity

def returnLengthOfTweet(tweet):

    return len(tweet.split())

def tokenizeTweet(tweet):

    return word_tokenize(tweet)


def removeStopwordsFromTweet(tweet):
    _ttweet = [tword for tword in tweet if not tword in stopwords.words()]

    return _ttweet, len(_ttweet)

def stemTweet(ttokens):

    tstems = []
    for ttoken in ttokens:
        tstems.append(_stemmer.stem(ttoken))

    return tstems, len(tstems)


def lemmatizeTweet(ttokens):

    tlemmas = []
    for ttoken in tlemmas:
        tlemmas.append(_lemmatizer.lemmatize(ttoken))

    return tlemmas, len(tlemmas)

def extractPartsOfSpeechFromTweet(ttokens):


    return nltk.pos_tag(ttokens)




def extractFeaturesFromPartsOfSpeech(partsOfSpeech):


    _partsOfSpeech = dict(partsOfSpeech)



    partsOfSpeechCount  = {}
    for key in _partsOfSpeech:

        if _partsOfSpeech[key] not in partsOfSpeechCount:
            partsOfSpeechCount[_partsOfSpeech[key]] = 1
        else:
            partsOfSpeechCount[_partsOfSpeech[key]] +=1


    for part in partsOfSpeechCount:
        partsOfSpeechCount[part] = round(partsOfSpeechCount[part]/len(_partsOfSpeech), 4)

    print(partsOfSpeechCount)

    return  partsOfSpeechCount


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

    print(_unigrams, _bigrams, _trigrams)
    return _unigrams, _bigrams, _trigrams




def preprocessTheTweet(tweet):
    print(tweet)

    # lowercase all the words in the tweet
    lowercasedTweet = lowercaseTheTweet(tweet)

    # replace the contracted words
    _lowercaseTweet = replaceContractedWords(lowercasedTweet)

    # remove rt, mentioned users, hashtags, urls, emojis and special characters from the tweet
    _lowercaseTweet = removeRTFromTweet(_lowercaseTweet)

    _lowercaseTweet = removeUserMentionsFromTweet(_lowercaseTweet)
    _lowercaseTweet = removeHashtagsFromTweet(_lowercaseTweet)
    _lowercaseTweet = removeURLsFromTweet(_lowercaseTweet)

    _lowercaseTweet = removeEmojisFromTweet(_lowercaseTweet)
    _lowercaseTweet = removeSpecialCharatctersFromTweet(_lowercaseTweet)

    _tokens, numberOfTokens = removeStopwordsFromTweet(tokenizeTweet(_lowercaseTweet))
    _stems, numberOfStems = stemTweet(_tokens)
    _lemmas, numberOfLemmas = lemmatizeTweet(_tokens)

    _partsOfSpeech = extractPartsOfSpeechFromTweet(_tokens)
    _partsOfSpeechFeatures = extractFeaturesFromPartsOfSpeech(_partsOfSpeech)

    _unigrams, _bigrams, _trigrams = extractNGramsFromTweet(_tokens)

    sentimentScore, sentiment = extractSentimentOfTweet(tweet)
    subjectivityScore, subjectivity = extractSubjectivityOfTweet(tweet)



    Tweet = {
        'original_tweet':tweet,
        'number_of_words': returnLengthOfTweet(tweet),

        'tokens': _tokens,
        'number_of_tokens': numberOfTokens,

        'stems' : _stems,
        'number_of_stems':numberOfStems,

        'lemmas': _lemmas,
        'number_of_lemmas': numberOfLemmas,

        'parts_of_speech':_partsOfSpeech,
        'parts_of_speech_features':_partsOfSpeechFeatures,

        'unigrams':_unigrams,
        'number_of_unigrams': len(_unigrams),
        'bigrams': _bigrams,
        'number_of_bigrams': len(_bigrams),
        'trigrams': _trigrams,
        'number_of_trigrams': len(_trigrams),

        'sentiment_score':sentimentScore,
        'sentiment': sentiment,
        'subjectivity_score': subjectivityScore,
        'subjectivity': subjectivity


    }


    print(_lowercaseTweet)
    return Tweet




if __name__=='__main__':
    tweet = 'RT @lm_stats: .@LittleMix’s ‘Heartbreak Anthem’ is now the longest charting song released this year by a female group on the Global Spotify…'
    Tweet = preprocessTheTweet(tweet)

    print(Tweet)







