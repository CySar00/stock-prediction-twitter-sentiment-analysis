import os,re,sys
import YahooFinance
import pandas as pd

from YahooFinance.ExtractStockInformation import *

COMPANIES = {
    'Google': 'GOOGL',
    'Apple': 'AAPL',
    'Amazon': 'AMZN',
    'Spotify': 'SPOT',
    'IBM': 'IBM',
    'Delta': 'DAL',
    'Pfizer': 'PFE',
    'Microsoft': 'MSFT'
}

COMPANY_CODES = [
    'GOOGL', 'AMZN', 'AAPL',
    'SPOT', 'MSFT', 'PFE',
    'DAL', 'IBM'
]


if __name__ =='__main__':

    for companyName, companyCode in COMPANIES.items():
        stockHistory = extractStockInformation(companyCode)

        pathToStockFile  =  './Data/StockInfo/'+ companyName + '.csv'
        stockHistory.to_csv(pathToStockFile, index=True,  header=True)
