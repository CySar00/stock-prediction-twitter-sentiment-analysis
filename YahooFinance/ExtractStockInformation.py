import os,re,sys
import yfinance
import math




def extractStockInformation(companyCode):

    stock = yfinance.Ticker(companyCode)
    historyData  = yfinance.download(companyCode, start="2021-02-01", end="2021-04-1")

    historyData['HighLoad'] = (( historyData['High'] - historyData['Close'])/historyData['Close']) * 100
    historyData['Change'] = ((historyData['Close'] - historyData['Open'])/historyData['Open']) * 100
    historyData['HighLow'] = ((historyData['High'] - historyData['Low'])/historyData['Low'])
    forecastCol = 'Close'
    forecastOut = int(math.ceil(0.01*len(historyData)))

    historyData['Label'] = historyData[[forecastCol]].shift(-forecastOut)

    return historyData




if __name__=='__main__':
    data  = extractStockInformation('SPOT')
    print(data)

