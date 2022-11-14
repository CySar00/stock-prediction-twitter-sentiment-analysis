import os,re,sys
import math

import matplotlib.pyplot as plt
import sklearn
import xgboost

import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn import preprocessing

from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor


from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor

import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM


root = './Data/TrainingAndTestSets/'
path_to_out_files = './Data/Figures/'

company_names = ['Amazon', 'Apple', 'Delta', 'Google', 'IBM', 'Microsoft', 'Pfizer', 'Spotify']


def mape(actual, pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100


def mse(actual, pred):
    return np.square(np.subtract(actual, pred)).mean()


def rmse(actual, pred):
    return math.sqrt(mse(actual, pred))







if __name__ == '__main__':
    #scaler = MinMaxScaler(feature_range=(0, 1))

    tf.random.set_seed(50)

    # scaler = MinMaxScaler(feature_range=(0, 1))
    lookback = 1000

    for company in company_names:
        _root = root + company
        print(company)

        path_to_training_set = _root + '/training_set.csv'
        path_to_test_set = _root + '/test_set.csv'

        training_set = pd.read_csv(path_to_training_set)
        test_set = pd.read_csv(path_to_test_set)

        n = len(test_set)

        x_train, y_train = training_set.iloc[:, :-2], training_set.iloc[:, -1]
        x_test, y_test = test_set.iloc[:, :-2], test_set.iloc[:, -1]


        x_train, y_train = x_train.to_numpy(), y_train.to_numpy()
        x_test, y_test= x_test.to_numpy(), y_test.to_numpy()

        lin_model = LinearRegression().fit(x_train, y_train)
        tree_model = DecisionTreeRegressor().fit(x_train, y_train)

        forest_model = RandomForestRegressor().fit(x_train, y_train)
        etree_model = ExtraTreesRegressor().fit(x_train, y_train)

        ada_model = AdaBoostRegressor().fit(x_train, y_train)
        grad_model = GradientBoostingRegressor().fit(x_train, y_train)

        xg_model = XGBRegressor().fit(x_train, y_train)

        svr_model = SVR(max_iter=1000000).fit(x_train, y_train)
        svr_lin_model = LinearSVR().fit(x_train, y_train)

        knn_model = KNeighborsRegressor(n_neighbors=2).fit(x_train, y_train)

        lin_yhat = lin_model.predict(x_test)

        tree_yhat = tree_model.predict(x_test)
        forest_yhat = forest_model.predict(x_test)
        etree_yhat = etree_model.predict(x_test)

        ada_yhat = ada_model.predict(x_test)
        grad_yhat = grad_model.predict(x_test)
        xg_yhat = xg_model.predict(x_test)

        svr_hat = svr_model.predict(x_test)
        svr_lin_hat = svr_lin_model.predict(x_test)

        knn_yhat = knn_model.predict(x_test)

        x_train1 = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))

        x_test1 = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        model = Sequential()
        model.add(LSTM(50, input_shape=(x_train1.shape[1], x_train1.shape[2])))
        model.add(Dense(100, activation='relu'))

        model.compile(loss='mean_absolute_error', optimizer='adam')

        history = model.fit(x_train1, y_train, epochs=50, batch_size=72, validation_data=(x_test1, y_test))

        rnn_yhat = model.predict(x_test1)



        test_set['lin_pred_label'] = lin_yhat
        test_set['decision_tree_label'] = tree_yhat
        test_set['random_forest_label'] = forest_yhat
        test_set['extra_tree_label'] = etree_yhat
        test_set['ada_boost_label'] = ada_yhat
        test_set['grad_boot_label'] = grad_yhat
        test_set['xg_boost_label'] = xg_yhat
        test_set['svr_label'] = svr_hat
        test_set['svr_lin_label'] = svr_lin_hat
        test_set['knn_label'] = knn_yhat

        mse_grad = mse(y_test, grad_yhat)
        rmse_grad = rmse(y_test, grad_yhat)
        mape_grad = mape(y_test, grad_yhat)

        mse_xg = mse(y_test, xg_yhat)
        rmse_xg = rmse(y_test, xg_yhat)
        mape_xg = mape(y_test, xg_yhat)

        mse_svr = mse(y_test, svr_hat)
        rmse_svr = rmse(y_test, svr_hat)
        mape_svr = mape(y_test,svr_hat)

        mse_svr_lin = mse(y_test, svr_lin_hat)
        rmse_svr_lin = rmse(y_test, svr_lin_hat)
        mape_svr_lin = mape(y_test, svr_lin_hat)

        mse_knn = mse(y_test, knn_yhat)
        rmse_knn = rmse(y_test, knn_yhat)
        mape_knn = mape(y_test, knn_yhat)

        mse_lin = mse(y_test, lin_yhat)
        rmse_lin = rmse(y_test, lin_yhat)
        mape_lin = mape(y_test, lin_yhat)

        mse_lin = mse(y_test, lin_yhat)
        rmse_lin = rmse(y_test, lin_yhat)
        mape_lin = mape(y_test, lin_yhat)

        mse_lin  = mse(y_test, lin_yhat)
        rmse_lin = rmse(y_test, lin_yhat)
        mape_lin = mape(y_test, lin_yhat)

        mse_tree  = mse(y_test, tree_yhat)
        rmse_tree = rmse(y_test, tree_yhat)
        mape_tree = mape(y_test,  tree_yhat)

        mse_forest  = mse(y_test, forest_yhat)
        rmse_forest = rmse(y_test, forest_yhat)
        mape_forest = mape(y_test, forest_yhat)

        mse_etree  = mse(y_test, etree_yhat)
        rmse_etree = rmse(y_test, etree_yhat)
        mape_etree = mape(y_test, etree_yhat)

        mse_ada  = mse(y_test, ada_yhat)
        rmse_ada = rmse(y_test, ada_yhat)
        mape_ada = mape(y_test, ada_yhat)


        print(mse_lin, rmse_lin, mape_lin)
        print(mse_tree, rmse_tree, mape_tree)
        print(mse_forest, rmse_forest, mape_forest)
        print(mse_etree, rmse_etree, mape_etree)
        print(mse_ada, rmse_ada, mape_ada)
        print(mse_xg, rmse_xg, mape_xg)

        print(mse_grad, rmse_grad, mape_grad)
        print(mse_svr, rmse_svr, mape_svr)
        print(mse_svr_lin, rmse_svr_lin, mape_svr_lin)
        print(mse_knn, rmse_knn, mape_knn)








        pathToSaveFig = path_to_out_files + company + '/'

        outputFile = path_to_out_files + 'test_set.csv'
        test_set.to_csv(outputFile, index = False)

        #pathToSaveFig = './Data/Figures/' + company + '/'

        """
        plt.plot(y_test)
        plt.title('Actual')
        plt.savefig(pathToSaveFig + 'Actual.png')
        plt.close()

        plt.plot(lin_yhat)
        plt.title('Linear Regression')
        plt.savefig(pathToSaveFig + 'Linear_Regression.png')
        plt.close()

        plt.plot(y_test, label = 'Actual')
        plt.plot(lin_yhat, label = 'Predicted')
        plt.title('Actual vs. Linear Regression')
        plt.savefig(pathToSaveFig + 'Actual_Linear_Regression.png')
        plt.close()

        plt.plot(tree_yhat)
        plt.title('Decision Tree')
        plt.savefig(pathToSaveFig + 'Decision_Regression.png')
        plt.close()

        plt.plot(y_test, label = 'Actual')
        plt.plot(tree_yhat, label = 'Predicted')
        plt.title('Actual vs. Decision Tree')
        plt.savefig(pathToSaveFig + 'Actual_Decision.png')
        plt.close()

        plt.plot(forest_yhat)
        plt.title('Random Forest')
        plt.savefig(pathToSaveFig + 'Random_Forest.png')
        plt.close()

        plt.plot(y_test)
        plt.plot(forest_yhat)
        plt.title('Actual vs. Random Forest')
        plt.savefig(pathToSaveFig + 'Actual_Forest.png')

        plt.close()

        plt.plot(etree_yhat)
        plt.title('Extra Tree')
        plt.savefig(pathToSaveFig + 'Extra Tree.png')
        plt.close()

        plt.plot(y_test )
        plt.plot(etree_yhat)
        plt.title('Actual vs. Extra Tree')
        plt.savefig(pathToSaveFig + 'Actual_Extra.png')
        plt.close()

        plt.plot(xg_yhat)
        plt.title('XG Boost')
        plt.savefig(pathToSaveFig + 'XGBoost.png')
        plt.close()

        plt.plot(y_test)
        plt.plot(tree_yhat)
        plt.title('Actual vs. XGBoost')
        plt.savefig(pathToSaveFig + 'Actual_XGBoost.png')
        plt.close()

        plt.plot(ada_yhat)
        plt.title('Ada Boost')
        plt.savefig(pathToSaveFig + 'Ada_Boost.png')
        plt.close()

        plt.plot(y_test)
        plt.plot(ada_yhat)
        plt.title('Actual vs. Ada Boost')
        plt.savefig(pathToSaveFig + 'Actual_vs_Ada_Boost.png')
        plt.close()

        plt.plot(grad_yhat)
        plt.title('Gradient Boost')
        plt.savefig(pathToSaveFig + 'Gradient_Boost.png')
        plt.close()

        plt.plot(y_test)
        plt.plot(grad_yhat)
        plt.title('Actual vs. Gradient Boost')
        plt.savefig(pathToSaveFig + 'Actual_Gradient_Boost.png')
        plt.close()

        plt.plot(svr_hat)
        plt.title('SVR')
        plt.savefig(pathToSaveFig + 'SVR.png')
        plt.close()

        plt.plot(y_test)
        plt.plot(svr_hat)
        plt.title('Actual vs. SVR')
        plt.savefig(pathToSaveFig + 'Actual_vs_SVR.png')
        plt.close()

        plt.plot(svr_lin_hat)
        plt.title('Linear SVR')
        plt.savefig(pathToSaveFig + 'Linear_SVR.png')
        plt.close()

        plt.plot(y_test)
        plt.plot(svr_lin_hat)
        plt.title('Actual vs. Linear SVR')
        plt.savefig(pathToSaveFig + 'Actual_vs_Linear_SVR.png')
        plt.close()

        plt.plot(knn_yhat)
        plt.title('KNN')
        plt.savefig(pathToSaveFig + 'KNN.png')
        plt.close()

        plt.plot(y_test)
        plt.plot(knn_yhat)
        plt.title('Actual vs. KNN')
        plt.savefig(pathToSaveFig + 'Actual_vs_KNN.png')
        plt.close()
        
        """

        figure, axes = plt.subplots(3, 4, figsize=(15, 15))

        axes[0, 0].plot(y_test, color = 'r', label = 'Actual')
        axes[0, 0].plot(lin_yhat, color = 'm', label = 'Predicted')
        axes[0, 0].set_title('Linear Regression')

        axes[0, 1].plot(y_test, color='r', label='Actual')
        axes[0, 1].plot(svr_hat, color='m', label='Predicted')
        axes[0, 1].set_title('SVR')

        axes[0, 2].plot(y_test, color='r', label='Actual')
        axes[0, 2].plot(svr_lin_hat, color='m', label='Predicted')
        axes[0, 2].set_title('Linear SVR')

        axes[0, 3].plot(y_test, color='r', label='Actual')
        axes[0, 3].plot(knn_yhat, color='m', label='Predicted')
        axes[0, 3].set_title('KNN')

        axes[1, 0].plot(y_test, color='r', label='Actual')
        axes[1, 0].plot(lin_yhat, color='m', label='Predicted')
        axes[1, 0].set_title('Decision Tree')

        axes[1, 1].plot(y_test, color='r', label='Actual')
        axes[1, 1].plot(tree_yhat, color='m', label='Predicted')
        axes[1, 1].set_title('Decision Tree')

        axes[1, 2].plot(y_test, color='r', label='Actual')
        axes[1, 2].plot(forest_yhat, color='m', label='Predicted')
        axes[1, 2].set_title('Random Forest')

        axes[1, 3].plot(y_test, color='r', label='Actual')
        axes[1, 3].plot(etree_yhat, color='m', label='Predicted')
        axes[1, 3].set_title('Extra Tree ')

        axes[2, 0].plot(y_test, color='r', label='Actual')
        axes[2, 0].plot(ada_yhat, color='m', label='Predicted')
        axes[2, 0].set_title('Ada Boost')

        axes[2, 1].plot(y_test, color='r', label='Actual')
        axes[2, 1].plot(grad_yhat, color='m', label='Predicted')
        axes[2, 1].set_title('Gradient Boost')

        axes[2, 2].plot(y_test, color='r', label='Actual')
        axes[2, 2].plot(xg_yhat, color='m', label='Predicted')
        axes[2, 2].set_title('XG Boost')

        axes[2, 3].plot(y_test, color='r', label='Actual')
        axes[2, 3].plot(rnn_yhat, color='m', label='Predicted')
        axes[2, 3].set_title('RNN with LSTMs')

        figure.savefig(pathToSaveFig + 'Results.png')



        lin_score = lin_model.score(x_test, y_test)


        tree_score = tree_model.score(x_test, y_test)
        forest_score = forest_model.score(x_test, y_test)
        etree_score = etree_model.score(x_test, y_test)

        ada_score = ada_model.score(x_test, y_test)
        grad_score = grad_model.score(x_test, y_test)

        knn_score = knn_model.score(x_test, y_test)











