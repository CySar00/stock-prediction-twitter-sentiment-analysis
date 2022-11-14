import os,re,sys
import sklearn
import math

import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import  KNeighborsRegressor

from xgboost import XGBRegressor

from sklearn.preprocessing import MinMaxScaler, StandardScaler


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM


companies = ['Amazon', 'Apple', 'Delta', 'Google', 'IBM', 'Microsoft', 'Pfizer', 'Spotify']


import matplotlib.colors as mcolors

root = './Data/TrainingAndTestSets/'
path_to_output_files = './Data/Figures/'


DICT  = {
    1: 'i', 2: 'ii', 3: 'iii',
    4: 'iv',  5:'v', 6: 'vi',
    7: 'vii', 8: 'viii', 9 : 'xi'
}

def mape(actual, pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100


def mse(actual, pred):
    return np.square(np.subtract(actual, pred)).mean()


def rmse(actual, pred):
    return math.sqrt(mse(actual, pred))


if __name__ == '__main__':
    print(1)

    i = 0

    sc1, sc2 = MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))
    seed = tf.random.set_seed(10)
    lookback = 1

    for compnay in companies:
        print(compnay)
        i+=1

        rroot = root + compnay

        path_to_training_set = rroot + '/training_set.csv'
        path_to_test_set = rroot + '/test_set.csv'

        output_dir = path_to_output_files + compnay + '/'

        training_set = pd.read_csv(path_to_training_set,  header=0)
        test_set = pd.read_csv(path_to_test_set, header= 0)

        dates = test_set.iloc[:, 0]

        n1 = len(training_set)
        n2 = len(test_set)

        x_train, y_train = training_set.iloc[:, 1:-2], training_set.iloc[:, -1]
        x_test, y_test = test_set.iloc[:, 1:-2],test_set.iloc[:, -1]



        X_train, Y_train =  np.array(x_train), np.array(y_train)
        X_test, Y_test = np.array(x_test), np.array( y_test)

        lin_model = LinearRegression().fit(X_train, Y_train)
        tree_model = DecisionTreeRegressor().fit(X_train, Y_train)
        forest_model = RandomForestRegressor().fit(X_train, Y_train)

        etree_model = ExtraTreesRegressor().fit(X_train, Y_train)
        ada_model = AdaBoostRegressor().fit(X_train, Y_train)
        grad_model = GradientBoostingRegressor().fit(X_train, Y_train)

        xg_model = XGBRegressor().fit(X_train, Y_train)
        
        lin_yhat = lin_model.predict(X_test)
        tree_yhat = tree_model.predict(X_test)
        forest_yhat = forest_model.predict(X_test)
        etree_yhat = etree_model.predict(X_test)
        ada_yhat = ada_model.predict(X_test)
        grad_yhat = grad_model.predict(X_test)

        xg_yhat = xg_model.predict(X_test)

        columns = training_set.columns

        y_train1 = y_train.values.reshape(-1, 1)

        xx_train = sc1.fit_transform(x_train)
        yy_train = sc2.fit_transform(y_train1)



        XX_train = []
        YY_train = []

        for j in range(lookback, n1):
            XX_train.append(xx_train[ j - lookback:j, :])
            YY_train.append(yy_train[j, 0] )

        XX_train1, YY_train1 = np.array(XX_train), np.array(YY_train)
        print(XX_train1.shape)

        XX_train2 = np.reshape(XX_train1, (XX_train1.shape[0], XX_train1.shape[1], XX_train1.shape[2]))

        xx_test = sc1.fit_transform(x_test)
        yy_test = sc2.fit_transform(y_test.values.reshape(-1, 1))


        XX_test = []
        YY_test = []
        for j in range(lookback, n2):
            XX_test.append(xx_test[ j - lookback : j, :])
            YY_test.append(yy_test[j, :])

        XX_test1 = np.array(XX_test)
        YY_test1 = np.array(YY_test)

        XX_test2 = np.reshape(XX_test1, (XX_test1.shape[0], XX_test1.shape[1], XX_test1.shape[2]))

        model = Sequential()
        model.add(LSTM(units = 50, return_sequences=True, input_shape=(XX_train2.shape[1], XX_train2.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units= 50))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss= 'mean_squared_error')

        history = model.fit(XX_train2, YY_train1, epochs=100, batch_size=72, validation_data=(XX_test2, YY_test1), verbose=4, shuffle=False)

        rnn_yhat = model.predict(XX_test2)

        RNN_yhat = sc2.inverse_transform(rnn_yhat)
        RNN_yhat1 = RNN_yhat[:,0]











        title = DICT[i] + '.' + compnay
        plt.figure(1, figsize= ( 16, 10))
        plt.plot(dates, y_test, color = 'xkcd:blue', label = 'Actual')
        plt.plot(dates, lin_yhat, color = 'xkcd:barney purple', label = 'Linear Regression')
        plt.plot(dates, tree_yhat, color = 'xkcd:dark pink', label = 'Decision Tree')
        plt.plot(dates, forest_yhat, color = 'xkcd:periwinkle', label = 'Random Forest')
        plt.plot(dates, etree_yhat, color = 'xkcd:royal purple', label = 'Extra Tree')
        plt.plot(dates, ada_yhat, color = 'xkcd:pale orange', label = 'Ada Boost')
        plt.plot(dates, grad_yhat, color = 'xkcd:green yellow', label = 'Gradient Boost')
        plt.plot(dates, xg_yhat, color = 'xkcd:crimson', label = 'XG Boost')


        plt.plot(dates[:len(RNN_yhat1)], RNN_yhat1, color = 'xkcd:dull green', label = 'RNN')
        plt.xlabel('Dates')
        plt.ylabel('Close Prices')
        plt.xticks(fontsize = 8)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(title, fontsize= 24)
        plt.savefig(output_dir + 'Results.png')
        plt.show()
        plt.close()













        mse_lin = mse(y_test, lin_yhat)
        rmse_lin = rmse(y_test, lin_yhat)
        mape_lin = mape(y_test, lin_yhat)

        print(mse_lin, rmse_lin, mape_lin)

        mse_tree = mse(y_test, tree_yhat)
        rmse_tree = rmse(y_test, tree_yhat)
        mape_tree = mape(y_test, tree_yhat)

        print(mse_tree, rmse_tree, mape_tree)

        mse_forest = mse(y_test, forest_yhat)
        rmse_forest = rmse(y_test, forest_yhat)
        mape_forest = mape(y_test, forest_yhat)

        print(mse_forest, rmse_forest, mape_forest)

        mse_etree = mse(y_test, etree_yhat)
        rmse_etree = rmse(y_test, etree_yhat)
        mape_etree = mape(y_test, etree_yhat)

        print(mse_etree, rmse_etree, mape_etree)

        mse_ada = mse(y_test, ada_yhat)
        rmse_ada = rmse(y_test, ada_yhat)
        mape_ada = mape(y_test, ada_yhat)

        print(mse_ada, rmse_ada, mape_ada)

        mse_grad = mse(y_test, grad_yhat)
        rmse_grad = rmse(y_test, grad_yhat)
        mape_grad = mape(y_test, grad_yhat)

        print(mse_grad, rmse_grad, mape_grad)

        mse_xg = mse(y_test, xg_yhat)
        rmse_xg = rmse(y_test, xg_yhat)
        mape_xg = mape(y_test, xg_yhat)

        print(mse_xg, rmse_xg, mape_xg)

        print(np.size(y_test[:len(RNN_yhat)]))
        print(np.size(RNN_yhat))

        mse_rnn = mse(y_test[:len(RNN_yhat1)], RNN_yhat1)
        rmse_rnn = rmse(y_test[:len(RNN_yhat1)], RNN_yhat1)
        mape_rnn = mape(y_test[:len(RNN_yhat1)], RNN_yhat1)

        print(mse_rnn, rmse_rnn, mape_rnn)





















