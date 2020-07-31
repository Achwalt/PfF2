#Meta--------------------------------------------------------------------------

#Benjamin Albrechts
#   Created: 29.06.2020 16:00
#   Purpose: ML-Tool for quick sentiments on cryptocurrency

#Google News API Alternatives:
#User: creacret@wegwerfemail.de
#PW: MAxxL6Ep2i3Em8L
#Key: add6f9d8cbc74b3b9dd029a0befaa5dc

#User: teloukit@wegwerfemail.de
#PW: tasUFaTq2eq2bV9
#Key: 99601741f1d249298d33d56c76de87b1

#User: thiachas@wegwerfemail.de
#PW: afi43tq344nasb24nbr
#Key: a4918ed23a2c44699f7dff8a0fb65a78

#Landing Page: 
#User: friovech@wegwerfemail.de
#PW: PZ3SPhaihUNG3j3
#Adresse: https://friovech.wixsite.com/website

#Import Section----------------------------------------------------------------

#Datawork
from pandas import *
import numpy as np

#API
import requests
from time import sleep

#ScikitLearn
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import linear_model, tree
#Machine Learning sein Vater
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.initializers import RandomUniform, RandomNormal

#statsmodels
import statsmodels.api as sm

#matplotlib
from matplotlib import pyplot

#Selenium
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait

#BeautifulSoup
from bs4 import BeautifulSoup


#Regular Expressions
import re

#Natural Language Toolkit
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import string

#Since this one is slow af
from tqdm import tqdm

#Deactivate warnings for overwriting copied slices in pandas dataframe
import warnings
warnings.filterwarnings("ignore")

#System time
from datetime import datetime, timedelta, timezone

#Make directories on the fly
import os
import winsound

#Make directory for report-----------------------------------------------------

try:
    os.mkdir("Output/Report")
except:
    pass

try:
    os.mkdir("Output/Report/Plots")
except:
    pass

try:
    os.mkdir("Output/Report/Tables")
except:
    pass

#Import the data---------------------------------------------------------------

#Filter out empty Textfiles - These arise from a timeout when downloading the HTML
def notempty(aVector):
    return([True if len(i) > 0 else False for i in aVector])

#Find duplicates in a list
def duplicated(aList):
    "Finds duplicated entries in a list and returns boolean index."
    aCopy = aList.copy()
    for i in range(len(aList)):
        if str(aCopy[i]) != "True":
            for j in range(len(aList)):
                if str(aList[i]) == str(aList[j]):
                    if i != j:
                        aCopy[i] = True
                        aCopy[j] = True
                else:
                    if j == len(aCopy) - 1:
                        aCopy[i] = False
    return aCopy

def boolInv(listOfBools):
    "Inverts list of booleans."
    return([not i for i in listOfBools])

def concString(aListOfStrings):
    String = ""
    for i in aListOfStrings:
        String += i + " "
    return String[:-1]

def liTxt(listOfStrings):
    "Compiles list of strings to string"
    stringo = ""
    for i in listOfStrings:
        stringo += str(i) + " "
    return(stringo[:-1])

def Xdrop(mydata, leave):
    "Drops every column from dataframe except what is supplied in list leave."
    try:
        to_drop = []
        for i in mydata.columns:
            if leave not in i:
                to_drop.append(i)
        return mydata.drop(to_drop, axis = 1)
    except TypeError:
        to_drop = []
        for k in leave:
            for i in mydata.columns:
                if k not in i:
                    to_drop.append(i)
        for k in leave:
            j = 0
            while j < len(to_drop):
                if k in to_drop[j]:
                    to_drop.remove(to_drop[j])
                else:
                    j += 1
        return mydata.drop(to_drop, axis = 1)
    
def dropX(mydata, kill):
    "Drops columns according to unique identifiers."
    try:
        to_drop = []
        for i in mydata.columns:
            if kill in i:
                to_drop.append(i)
        return mydata.drop(to_drop, axis = 1)
    except TypeError:
        to_drop = []
        for k in kill:
            for i in mydata.columns:
                if k in i:
                    to_drop.append(i)
        return mydata.drop(to_drop, axis = 1)
    
#Find duplicated Articles
#It is unlikely that we can classify an article which is relevant for a multitude of cryptocurrencies.
#While there is FOMO, there are also reasons why one rises and another falls.
#Thus, I must filter them out.
#GNews = GNews[boolInv(duplicated(list(GNews["Fulltext"])))]
#GNews.index = range(len(GNews["Fulltext"]))


#StockTwits needs no Machine learning and anyways tests my patience with webscraping way tooooooooo much
#StockTwits = read_csv(r"Output/StockTwits.csv", sep = ";", index_col = 0)

#Import Dictionary-------------------------------------------------------------
wordLibrary = {"Positive":[], "Negative":[]}
with open("Dependencies/negative-words.txt", "r") as f:
    for line in f:
        if len(line)>0 and line[0:1] != ";":
            wordLibrary["Negative"].append(line.replace("\n",""))
f.close()
           
with open("Dependencies/positive-words.txt", "r") as f:
    for line in f:
        if len(line)>0 and line[0:1] != ";":
            wordLibrary["Positive"].append(line.replace("\n",""))
f.close()
            
del wordLibrary["Positive"][0], wordLibrary["Negative"][0]

#Import GNews------------------------------------------------------------------
Gnews = read_csv("Output/Available News.csv", sep = ";", encoding = "utf-8-sig", index_col = 0)
Gnews.index = to_datetime(Gnews["Release"])
Gnews["Fulltext"] = np.NaN
for i in range(len(Gnews.index)):
    fname = "Output/News - Raw Text/"+re.sub("["+string.punctuation+"'`´’']", "", Gnews["Headline"][i]) + ".txt"
    with open("Output/News - Raw Text/"+re.sub("["+string.punctuation+"'`´’']", "", Gnews["Headline"][i]) + ".txt", "r", encoding = "utf-8-sig") as file:
        Gnews["Fulltext"][Gnews.index[i]] = file.read().replace("\n","")
    file.close()

Gnews = Gnews[Gnews.index != NaT]
#StockTwits = read_csv("Dependencies/StockTwits.csv", sep = ";", encoding = "utf-8-sig")

#Extracting Features: TFxIDF---------------------------------------------------
wordFreq = {}
docFreq = {}
for i in wordLibrary["Negative"]:
    wordFreq[i] = []
for i in wordLibrary["Positive"]:
    wordFreq[i] = []

for i in wordLibrary["Negative"]:
    docFreq[i] = []
for i in wordLibrary["Positive"]:
    docFreq[i] = []

sentWords = {"Positive":[], "Negative":[]}

counter = -1
for i in Gnews.index:
    counter += 1
    for key in wordFreq.keys():
        wordFreq[key].append(0)
    sourceText = str(Gnews["Headline"][i]) + " " + str(Gnews["Summary"][i]) + " " + str(Gnews["LeadParagraph"][i]) + " " + str(Gnews["Fulltext"][i])
    #Also creates lowercases
    text = nltk.sent_tokenize(sourceText)
    #Remove Interpunctuation
    for k in range(len(text)):
        text[k] = text[k].lower()
        text[k] = re.sub(r"\W", " ", text[k])
        text[k] = re.sub(r"\s+", " ", text[k])
    #Sum up WordFreq
    posWords = 0
    negWords = 0
    totWords = 0
    for sentence in text:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token in wordFreq.keys():
                wordFreq[token][counter] += 1
            if token in wordLibrary["Negative"]:
                negWords += 1
            if token in wordLibrary["Positive"]:
                posWords += 1
            totWords += 1
    
    sentWords["Positive"].append(posWords/(totWords+1))
    sentWords["Negative"].append(negWords/(totWords+1))
    
    #Get Document Frequ
    for key in wordFreq.keys():
        if key in sourceText:
            docFreq[key].append(1)
        else:
            docFreq[key].append(0)

#Convert to dataframe for consistency
wordFreq = DataFrame(wordFreq)
docFreq = DataFrame(docFreq)
sentWords = DataFrame(sentWords)

#Get inverse doc frequency
invDocFreq = {}
for col in docFreq.columns:
    invDocFreq[col] = np.log(len(docFreq[col])/(1+sum(docFreq[col])))
invDocFreq = DataFrame(invDocFreq, index = [0])

#Term Frequency X Inverse Document Frequency as applied features
tfXidf = DataFrame(wordFreq.values * invDocFreq.values)
tfXidf.index = wordFreq.index
tfXidf.columns = wordFreq.columns

#Features
GnFeat = sentWords.join(tfXidf)
#Scale
xSS = StandardScaler()
xSS.fit(GnFeat)
xTrain = xSS.transform(GnFeat)

model = load_model('Dependencies/News Predictor/model.h5')
#In One-Hot-Encoding Format
sentiments = DataFrame(model.predict(GnFeat))
for i in sentiments.columns:
    sentiments[i] = np.where(sentiments[i] == sentiments.max(axis = 1), 1, 0)
Gnews["Sentiment"] = np.where(sentiments[0] == 1, -1, np.where(sentiments[1] == 1, 0, np.where(sentiments[2] == 1, 1, np.NaN)))

#Import StockTweets------------------------------------------------------------
stockTwits = read_csv("Output/StockTwits.csv", sep =";", encoding = "utf-8-sig", index_col = "Created")
stockTwits["SentimentEnc"] = np.where(stockTwits["Sentiment"] == "Bullish", 1, np.where(stockTwits["Sentiment"] == "Bearish", -1, np.NaN))
stockTwits.index = to_datetime(stockTwits.index).tz_localize(tz = 0)

#Import Prices-----------------------------------------------------------------
coins = []
for filename in os.listdir("Output/Prices"):
    try:
        f = read_csv("Output/Prices/" + filename, sep = ";", encoding = "utf-8-sig", index_col = 0)
        f.index = to_datetime(f.index)
        f.columns = array(f.columns)+" "+str(filename)[:-4]
        prices = merge(prices, f, left_index = True, right_index = True)
        coins.append(str(filename)[:-4])
    except:
        f = read_csv("Output/Prices/" + filename, sep = ";", encoding = "utf-8-sig", index_col = 0)
        f.index = to_datetime(f.index)
        f.columns = array(f.columns)+" "+str(filename)[:-4]
        prices = f
        coins.append(str(filename)[:-4])

#Output Charts-----------------------------------------------------------------

#Prices
#Full Time Horizon
for i in coins:
    plotData = Xdrop(prices, i)
    plotData = dropX(plotData, ["Volume", "Market"])
    pyplot.figure()
    pyplot.xticks(rotation = 20)
    trendline = list(sm.OLS(Xdrop(plotData, "Close"), sm.add_constant(list(range(len(plotData.index))))).fit().predict(sm.add_constant(list(range(len(plotData.index))))))[::-1]
    plotData["Trend"] = list(trendline)[::-1]
    pyplot.plot(plotData)
    pyplot.title("Prices: " + i.capitalize())
    pyplot.legend(plotData.columns)
    pyplot.savefig("Output/Report/Plots/PricesAllData - " + i.capitalize(), dpi = 400, quality = 100)

#For the last 28 days
for i in coins:
    plotData = Xdrop(prices, i)
    plotData = dropX(plotData, ["Volume", "Market"])
    plotData = plotData[plotData.index >= (plotData.index[0]-timedelta(28))]
    pyplot.figure()
    pyplot.xticks(rotation = 20)
    trendline = list(sm.OLS(Xdrop(plotData, "Close"), sm.add_constant(list(range(len(plotData.index))))).fit().predict(sm.add_constant(list(range(len(plotData.index))))))[::-1]
    plotData["Trend"] = list(trendline)[::-1]
    pyplot.plot(plotData)
    pyplot.title("Prices 28 Days: " + i.capitalize())
    pyplot.legend(plotData.columns)
    pyplot.savefig("Output/Report/Plots/Prices28Days - " + i.capitalize(), dpi = 400, quality = 100)
    
#For the last 7 days
for i in coins:
    plotData = Xdrop(prices, i)
    plotData = dropX(plotData, ["Volume", "Market"])
    plotData = plotData[plotData.index >= (plotData.index[0]-timedelta(7))]
    pyplot.figure()
    pyplot.xticks(rotation = 20)
    trendline = list(sm.OLS(Xdrop(plotData, "Close"), sm.add_constant(list(range(len(plotData.index))))).fit().predict(sm.add_constant(list(range(len(plotData.index))))))[::-1]
    plotData["Trend"] = list(trendline)[::-1]
    pyplot.plot(plotData)
    pyplot.title("Prices 7 Days: " + i.capitalize())
    pyplot.legend(plotData.columns)
    pyplot.savefig("Output/Report/Plots/Prices7Days - " + i.capitalize(), dpi = 400, quality = 100)
    
#Market Caps ------------------------------------------------------------------
#Full Time Horizon
plotData = Xdrop(prices, "Market")
#plotData = plotData[plotData.index >= (plotData.index[0]-timedelta(0))]
pyplot.figure()
pyplot.xticks(rotation = 20)
pyplot.plot(DataFrame(plotData.values/plotData[plotData.index == plotData.index[-1]].values, index = plotData.index))
pyplot.title("MarketCap: " + i.capitalize())
pyplot.legend(plotData.columns)
pyplot.savefig("Output/Report/Plots/MarketCapsAllData", dpi = 400, quality = 100)

#Last 28 Days
plotData = Xdrop(prices, "Market")
plotData = plotData[plotData.index >= (plotData.index[0]-timedelta(28))]
pyplot.figure()
pyplot.xticks(rotation = 20)
pyplot.plot(DataFrame(plotData.values/plotData[plotData.index == plotData.index[-1]].values, index = plotData.index))
pyplot.title("MarketCap 28 Days: " + i.capitalize())
pyplot.legend(plotData.columns)
pyplot.savefig("Output/Report/Plots/MarketCaps28Days", dpi = 400, quality = 100)

#Last 7 Days
plotData = Xdrop(prices, "Market")
plotData = plotData[plotData.index >= (plotData.index[0]-timedelta(7))]
pyplot.figure()
pyplot.xticks(rotation = 20)
pyplot.plot(DataFrame(plotData.values/plotData[plotData.index == plotData.index[-1]].values, index = plotData.index))
pyplot.title("MarketCap 7 Days: " + i.capitalize())
pyplot.legend(plotData.columns)
pyplot.savefig("Output/Report/Plots/MarketCaps7Days", dpi = 400, quality = 100)

#Volumes ------------------------------------------------------------------
#Full Time Horizon
plotData = Xdrop(prices, "Market")
#plotData = plotData[plotData.index >= (plotData.index[0]-timedelta(0))]
pyplot.figure()
pyplot.xticks(rotation = 20)
pyplot.plot(DataFrame(plotData.values/plotData[plotData.index == plotData.index[-1]].values, index = plotData.index))
pyplot.title("Traded Volume: " + i.capitalize())
pyplot.legend(plotData.columns)
pyplot.savefig("Output/Report/Plots/VolumesAllData", dpi = 400, quality = 100)

#Last 28 Days
plotData = Xdrop(prices, "Market")
plotData = plotData[plotData.index >= (plotData.index[0]-timedelta(28))]
pyplot.figure()
pyplot.xticks(rotation = 20)
pyplot.plot(DataFrame(plotData.values/plotData[plotData.index == plotData.index[-1]].values, index = plotData.index))
pyplot.title("Traded Volume 28 Days: " + i.capitalize())
pyplot.legend(plotData.columns)
pyplot.savefig("Output/Report/Plots/Volumes28Days", dpi = 400, quality = 100)

#Last 7 Days
plotData = Xdrop(prices, "Market")
plotData = plotData[plotData.index >= (plotData.index[0]-timedelta(7))]
pyplot.figure()
pyplot.xticks(rotation = 20)
pyplot.plot(DataFrame(plotData.values/plotData[plotData.index == plotData.index[-1]].values, index = plotData.index))
pyplot.title("Traded Volume 7 Days: " + i.capitalize())
pyplot.legend(plotData.columns)
pyplot.savefig("Output/Report/Plots/Volumes7Days", dpi = 400, quality = 100)

#GNewsSentiments---------------------------------------------------------------

for i in coins:
    plotData = Xdrop(prices, i)
    plotData = Xdrop(plotData, "Close")
    plotData = plotData[plotData.index >= (plotData.index[0]-timedelta(28))]
    plotData = DataFrame(plotData.values/plotData[plotData.index == plotData.index[-1]].values, index = plotData.index, columns = plotData.columns)
    plotData.index = plotData.index.tz_localize(tz = 0)
    plotData = merge(plotData, Gnews["Sentiment"][Gnews["Coin"] == i].resample("D").mean(), left_index = True, right_index = True)
    pyplot.figure()
    pyplot.xticks(rotation = 20)
    pyplot.plot(plotData)
    pyplot.title("Prices vs. GNews Sentiment: " + i.capitalize())
    pyplot.legend(plotData.columns)
    pyplot.savefig("Output/Report/Plots/PriceGNewsSentiment - " + i.capitalize(), dpi = 400, quality = 100)
    
#StockTwits Sentiments---------------------------------------------------------

stockTwitsMapping = {"bitcoin":["BTC.X", "BCH.X"], "ethereum":["ETH.X"], "eos":["EOS.X"]}

def within(anArray, aDict, aKey):
    asdf = []
    for i in anArray:
        if i in aDict[aKey]:
            asdf.append(True)
        else:
            asdf.append(False)
    return(asdf)

for i in coins:
    plotData = Xdrop(prices, i)
    plotData = Xdrop(plotData, "Close")
    plotData = plotData[plotData.index >= (plotData.index[0]-timedelta(28))]
    plotData = DataFrame(plotData.values/plotData[plotData.index == plotData.index[-1]].values, index = plotData.index, columns = plotData.columns)
    plotData.index = plotData.index.tz_localize(tz = 0)
    plotData = merge(plotData, stockTwits["SentimentEnc"][within(stockTwits["CoinIter"], stockTwitsMapping, i)].dropna().resample("D").mean(), left_index = True, right_index = True)
    pyplot.figure()
    pyplot.xticks(rotation = 20)
    pyplot.plot(plotData)
    pyplot.title("Prices vs. StockTwits Sentiment: " + i.capitalize())
    pyplot.legend(plotData.columns)
    pyplot.savefig("Output/Report/Plots/PriceStockTwitsSentiment - " + i.capitalize(), dpi = 400, quality = 100)
    
DataFrame(round(prices[prices.index >= (prices.index[0]-timedelta(28))].describe(), 2)).to_html("Output/Report/Tables/Prices28Days.html")
#Dataframe is in the "wrong" direction => shift -1
DataFrame(round(DataFrame(np.log((prices[prices.index >= (prices.index[0]-timedelta(28))].values)/(prices[prices.index >= (prices.index[0]-timedelta(28))].shift(-1).values)*365), index = prices[prices.index >= (prices.index[0]-timedelta(28))].index, columns = prices[prices.index >= (prices.index[0]-timedelta(28))].columns).describe(), 4)).to_html("Output/Report/Tables/LogRets28Days.html")