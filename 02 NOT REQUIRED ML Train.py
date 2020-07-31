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
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.initializers import RandomUniform, RandomNormal

import tensorflow as tf

#matplotlib
import matplotlib.pyplot as plt

#Selenium
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException

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

#Some functions----------------------------------------------------------------
def symbolsToString(alist, sep = " "):
    "Takes soup.findAll object and converts it to a separated string"
    astring = ""
    for i in alist:
        if re.search("\$[A-Z]",i.text) != None:
            astring += i.text + sep
    return astring

def repeat(X, nTimes):
    "Makes a n-lengthed list of x"
    retList = []
    for i in range(nTimes):
        retList.append(X)
    return retList

def liTxt(listOfStrings):
    "Compiles list of strings to string"
    stringo = ""
    for i in listOfStrings:
        stringo += str(i) + " "
    return(stringo[:-1])

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

#Import curated Dataset--------------------------------------------------------
Gnews = read_csv("Dependencies/MLTrainingData.csv", sep = ";", encoding = "utf-8-sig")
Gnews.index = to_datetime(Gnews["Release"])
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

#Encode Features
ySent = Gnews["Sentiment"]
ySent = np.where(ySent == "Positive", 2, np.where(ySent == "Negative", 0, np.where(ySent == "Neutral", 1, 666)))

#TF*IDF------------------------------------------------------------------------

xTrain, xTest, yTrain, yTest = train_test_split(sentWords.join(tfXidf)[ySent != 666],ySent[ySent != 666], test_size = 0.2, random_state = 123)

#Converting to numpy means we loose the rows and cols
xTrain_cols = xTrain.columns
xTrain_ind = xTrain.index

#Scale
xSS = StandardScaler()
xSS.fit(xTrain)

xTrain = xSS.transform(xTrain)
xTest = xSS.transform(xTest)

#Re-attach rows and cols
#xTrain.columns = xTrain_cols
#xTrain.index = xTrain_ind
#del xTrain_ind, xTrain_cols
"""
#Train the model using Keras Neural Network incl. hidden layer-----------------
model = Sequential()
#initWeights = RandomUniform(minval = -1, maxval = 1, seed = 1000)
#initWeights = RandomNormal(mean = 0, stddev = 1, seed = 1000)
model.add(Dense(len(docFreq.columns), activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(int(len(docFreq.columns)/10), activation = "sigmoid"))
#model.add(Dropout(0.25))
#model.add(Dense(int(len(docFreq.columns)/100), activation = "relu"))
model.add(Dense(3, activation = "softmax"))
#Use stochastic gradient descent with momentum as the optimizer
opt = SGD(lr = 0.001, momentum = 0.1)
model.compile(optimizer = opt, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

fitted = model.fit(xTrain, yTrain, epochs=100)
#Examination time
loss_, acc_ = model.evaluate(xTest, yTest)

evalSet = {"Predictions":model.predict(xTest), "Actuals":yTest}

#Examination time
loss_, acc_ = model.evaluate(xTest, yTest)

print("______________________________________________________________________\n\n\tModel TF*IDF out of sample:", acc_,"\n______________________________________________________________________")
"""

#####################################################
# We know now after evaluation that the model works #
# => Train it on 100% of the data!                  #
#####################################################

#Train the model using Keras Neural Network incl. hidden layer-----------------
model = Sequential()
model.add(Dense(len(docFreq.columns), activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(int(len(docFreq.columns)/10), activation = "sigmoid"))
model.add(Dense(3, activation = "softmax"))
#Use stochastic gradient descent with momentum as the optimizer
opt = SGD(lr = 0.001, momentum = 0.1)
model.compile(optimizer = opt, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

fitted = model.fit(sentWords.join(tfXidf)[ySent != 666], ySent[ySent != 666], epochs=1500)

model.save('Dependencies/News Predictor/model.h5')

"""
#Missing Words: Dictionary Improvements----------------------------------------

wordsAvailable = list(tfXidf.columns)
everything = liTxt(Gnews["Fulltext"])
sentInText = nltk.sent_tokenize(everything)
wordsInText = []
#Remove Interpunctuation
for k in range(len(sentInText)):
    sentInText[k] = sentInText[k].lower()
    sentInText[k] = re.sub(r"\W", " ", sentInText[k])
    sentInText[k] = re.sub(r"\s+", " ", sentInText[k])
    wordsInText.extend(word_tokenize(sentInText[k]))
#Don't keep multiple entries
wordsInText = list(set(wordsInText))

missingWords = []
i = 0
for i in wordsInText:
    if i not in wordsAvailable:
        missingWords.append(i)
    
missingWords = [i for i in missingWords if i not in stopwords.words("english")]

with open("Dependencies/supplementary_words.txt", "w", encoding = "utf-8-sig") as f:
    for i in missingWords:
        f.write(str(i) + "\n")
f.close
"""