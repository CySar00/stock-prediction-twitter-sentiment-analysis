import os,re,sys


import pandas as pd
import matplotlib.pyplot as plt

companies = ['Amazon', 'Apple', 'Delta', 'Google', 'IBM', 'Microsoft', 'Pfizer', 'Spotify']

root = './Data/TrainingAndTestSets/'
path_to_output_files = './Data/Figures/'

DICT  = {
    1: 'i', 2: 'ii', 3: 'iii',
    4: 'iv',  5:'v', 6: 'vi',
    7: 'vii', 8: 'viii', 9 : 'xi'
}

if __name__=='__main__':

    i = 0
    for compnay in companies:
        print(compnay)
        i +=1
        rroot = root + compnay

        path_to_training_set = rroot + '/training_set.csv'
        path_to_test_set = rroot + '/test_set.csv'

        training_set = pd.read_csv(path_to_training_set)
        test_set = pd.read_csv(path_to_test_set)

        output_dir = path_to_output_files + compnay + '/'



        emotions = test_set[['Date', 'emotion_score']]
        print(emotions['Date'])
        emotions1 = emotions.groupby('Date')

        subjectivity = test_set[['Date', 'subjectivity_score']]
        subjectivity1 = subjectivity.groupby('Date')

        print(test_set.columns)

        #groupedEmotions = emotions.groupby('Date')
        #groupedSubjectivity = subjectivity.groupby('Date')


        title = DICT[i] + '. ' + compnay


        plt.figure(1)
        plt.plot(emotions1['emotion_score'].sum(), color = 'r')
        plt.title(title)
        plt.ylabel('Emotion Score')
        plt.xlabel('Date')
        plt.xticks(fontsize = 8)
        plt.savefig(output_dir + 'Emotions.png')

        plt.close()

        plt.figure(2)
        plt.plot(subjectivity1['subjectivity_score'].sum(), color='r')
        plt.title(title)
        plt.ylabel('Subjectivity Score')
        plt.xlabel('Date')
        plt.xticks(fontsize=8)
        plt.savefig(output_dir + 'Subjectivity.png')

        plt.close()






