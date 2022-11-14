import os,re,sys
import glob
import math


import pandas as pd


root = './Data/NGrams/*.csv'
pathToCSVFiles = './Data/TrainingAndTestSets/'

TRAINING_SIZE  = 0.85

from ToBeRemoved import *




if __name__ == '__main__':

    for csvFile in glob.glob(root):
        company = csvFile.split('\\')[-1]
        company = company.split('.')[0]
        print(company)

        dataFrame = pd.read_csv(csvFile, low_memory=False)

        number_of_training_samples = math.ceil(TRAINING_SIZE*len(dataFrame))
        number_of_test_samples =len(dataFrame) - number_of_training_samples

        dataFrame.sort_values('created_at')

        data_frame1 = dataFrame[ ADD + [c for c in dataFrame.columns if c not in TO_BE_REMOVED and c not in STOCKS] + STOCKS]
        data_frame2 = data_frame1[[c for c in data_frame1.columns if c not in CLOSE] + CLOSE]

        print(data_frame2.columns)

        data_frame2.sort_values('Date')




        #training_set = data_frame2.iloc[:number_of_training_samples, :]
        #test_set = data_frame2.iloc[number_of_training_samples+1:, :]

        data_frame3 = data_frame2.sort_values(by='Date', ascending=True)

        mask1 = data_frame3['Date'] < '2021-03-23'
        training_set = data_frame3.loc[mask1]

        """"
        n1 = len(training_set)

        globalEmotion = 0
        GlobalEmotion = []
        for i, ttrow  in training_set.iterrows():
            globalEmotion += ttrow['emotion_score']

            GlobalEmotion.append(globalEmotion)

        GlobalEmotion = [i/n1 for i in GlobalEmotion]

        print(len(GlobalEmotion))
        print(len(training_set))
        training_set.loc[:, 'global_emotion_score'] = GlobalEmotion
        """

        mask2 = data_frame3['Date'] >= '2021-03-23'

        test_set1 = data_frame3.loc[mask2]
        mask3 = data_frame1['Date'] <= '2021-03-31'
        test_set = test_set1.loc[mask3]

        n1 = len(test_set)

        globalEmotion = 0

        """
        GlobalEmotion = []
        for i, trow in test_set.iterrows():
            globalEmotion += trow['emotion_score']

            GlobalEmotion.append(globalEmotion)

        GlobalEmotion = [i / n1 for i in GlobalEmotion]
    
        print(len(GlobalEmotion))
        print(len(test_set))

        test_set.loc[:, 'global_emotion_score'] = GlobalEmotion
        print(test_set['global_emotion_score'])
        
        test_set.fillna(0)
        training_set.fillna(0)
        print(training_set['global_emotion_score'])
        """


        path_to_training_set = pathToCSVFiles + company + '/'+ 'training_set.csv'
        path_to_test_set = pathToCSVFiles + company + '/' + 'test_set.csv'


        training_set.to_csv(path_to_training_set, index=False, header=True)
        test_set.to_csv(path_to_test_set, index=False, header=True)














