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


#Make directories--------------------------------------------------------------

try:
    os.mkdir("Output")
except:
    pass
try:
    os.mkdir("Output/News - Raw Text")
except:
    pass
try:
    os.mkdir("Output/News - HTML Library")
except:
    pass
try:
    os.mkdir("Output/Prices")
except:
    pass

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

#Acquire Data------------------------------------------------------------------

print("\n_ _ _ _ _ _ ________________________________________________________\n\n\tCoded by: Benjamin Albrechts\n\tVersion: 0.1\n\tFree Software. No guarantees for accuracy.\n\n\t***DISCLAIMER:***\n\tDo not blame me for your decision making!\n\tThe informative value of this tool is ZERO.\n________________________________________________________ _ _ _ _ _ _\n")

forXDays = 28

driver = webdriver.Edge("msedgedriver.exe")
#driver.get("https://friovech.wixsite.com/website")

#Your API key is:
API_Key = "add6f9d8cbc74b3b9dd029a0befaa5dc"
#Alternative
#API_Key = "99601741f1d249298d33d56c76de87b1"
#Alternative 2
#API_Key = "a4918ed23a2c44699f7dff8a0fb65a78"

#Collect data via GNews API using this URL:

#Google Search Keys for Cryptocurrencies:
coins = ["bitcoin", "ethereum", "eos"]

#Can only process english sources
sources = "ca,us,au,gb,nz"

#Select timeframe: One month max

newsdb = {}

for coin in coins:
    #Trying to be smarter than Google...
    for day_ in tqdm(range(forXDays), desc = "Obtaining news index for " + coin.capitalize(), leave = False):
        start_ = str((to_datetime(datetime.today()-timedelta(day_)).isoformat()))
        stop_ = str((to_datetime(datetime.today()-timedelta(day_-1)).isoformat()))
      
        url = "https://newsapi.org/v2/everything?q=crypto AND " + coin +  " AND (sentiment OR bearish OR bullish)" + "&source=" + sources + "&from=" + start_ + "&to=" + stop_ + "&apiKey=" + API_Key
        
        try:
            content = list(requests.get(url).json()["articles"])
        #We have made too many requests
        except KeyError:
            while True:
                input("ERROR: Too many requests were made, the app is therefore terminated. Please close the application and try again later!" )
                print("\n\t-> You cannot proceed from here.")
        #Dynamically create a dictionary of contents
        try:
            newsdb[coin].extend(content)
        except:
            newsdb[coin] = content

del coin, sources, day_, start_, stop_, url

#Create a processable Index File
newsdb_ind = {"Coin":[],"Headline":[],"Summary":[],"LeadParagraph":[],"Release":[],"Publisher":[],"Author":[],"URL":[]}

for coin in coins:
    for i in tqdm(newsdb[coin], desc = "Compiling Index for " + coin.capitalize(), leave = False):
        newsdb_ind["Coin"].append(coin)
        newsdb_ind["Headline"].append(i["title"])
        newsdb_ind["Summary"].append(i["description"])
        newsdb_ind["LeadParagraph"].append(i["content"])
        newsdb_ind["Release"].append(i["publishedAt"])
        newsdb_ind["Publisher"].append(i["source"]["name"])
        newsdb_ind["Author"].append(i["author"])
        newsdb_ind["URL"].append(i["url"])

#Save as usable CSV
DataFrame(newsdb_ind).to_csv("Output/Available News.csv", sep = ";", encoding = "utf-8")


#Import fulltext of all News---------------------------------------------------

#There is no way to retrieve the original text by Google API.
#Fake being a browser
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0'}

newsdb_ind["Fulltext"] = []
j = -1
refused_connections = 0
for i in tqdm(newsdb_ind["URL"], desc = "Downloading Fulltext of all News", leave = False):
    j += 1
    
    if j > 0:
        if newsdb_ind["Publisher"][j] == newsdb_ind["Publisher"][j-1]:
            sleep(1)
    
    try:
        text_ = requests.get(i, headers = headers).text
        
        if re.search("I am not a robot", text_, re.IGNORECASE) != None:
            driver.get(i)
            text_ = driver.page_source
            while re.search("I am not a robot", text_, re.IGNORECASE) != None:
                winsound.PlaySound("Dependencies/warning.wav", winsound.SND_FILENAME)
                override = input("Solve the 'I am not a robot' puzzle and confirm by clicking ENTER. Enter 'x' to override. ")
                if override == "x":
                    break
                else:
                    text_ = driver.page_source
    except requests.exceptions.ConnectionError:
        refused_connections += 1
        driver.get(i)
        text_ = driver.page_source
        while re.search("I am not a robot", text_, re.IGNORECASE) or re.search("recaptcha", text_, re.IGNORECASE) != None:
            winsound.PlaySound("Dependencies/warning.wav", winsound.SND_FILENAME)
            override = input("Solve the 'I am not a robot' puzzle and confirm by clicking ENTER. Enter 'x' to override. ")
            if override == "x":
                break
            else:
                text_ = driver.page_source
        
        
    #Write to file
    file = open("Output/News - HTML Library/"+re.sub("["+string.punctuation+"'`´’']", "", newsdb_ind["Headline"][j]) + ".html", "w", encoding = "utf-8")
    file.write(text_)
    file.close
    
    soup = BeautifulSoup(text_)
    
    for script in soup(["script", "style"]):
        script.decompose() 
    
    #Trying to keep text only
    plaintext = soup.get_text()
    
    #This is still way too much noise => Try to identify long sentences for some quality at least...
    all_sentences = re.findall(r'([A-Z][^\.!?]*[\.!?])', plaintext)
    c = -1
    for i in all_sentences:
        c += 1
        words = i.split()
        if len(words) > 0:#Error handle for empty word
            stopwords_ = sum([1 for word in words if word in stopwords.words("english")])
            #Sometimes trash is getting mixed up and falls below the radar, like: Subscribe NowLogin hereGet your free trial
            #Throw it out then
            for word in words:
                if re.search("[a-z][A-Z]", word) != None:
                    stopwords_ = 0
            if stopwords_/len(words) < 0.1:
                all_sentences[c] = ""
            
    #Put back to file
    plaintext = ""
    for i in all_sentences:
        for word in i.split():
            plaintext += word + " "
    #Wow, the quality suddenly improved dramatically :O Figuring that sh!t out took longer than it might look...
    #Not perfect, but certainly lightyears better than the crap that was outputted by beautifulsoup beforehand
    
    #Write to file
    file = open("Output/News - Raw Text/"+re.sub("["+string.punctuation+"'`´’']", "", newsdb_ind["Headline"][j]) + ".txt", "w", encoding = "utf-8")
    file.write(plaintext)
    file.close
    
    newsdb_ind["Fulltext"].append(plaintext)

if refused_connections > 0:
    print("Connections actively refused by server: "+str(refused_connections))


#Import Prices from Coinmarketcap.com------------------------------------------

stop_ = str(to_datetime(datetime.today()))[0:10].replace("-","")
start_ = str(to_datetime(datetime.today())-timedelta(2*forXDays))[0:10].replace("-","")

for coin in tqdm(coins, desc = "Downloading Prices from CoinMarketCap", leave = False):
    
    url = "https://coinmarketcap.com/currencies/" + coin + "/historical-data/?start=" + start_ + "&end=" + stop_
    
    try:
        text_ = requests.get(url, headers = headers).text
        
        if re.search("I am not a robot", text_, re.IGNORECASE) != None:
            driver.get(url)
            text_ = driver.page_source
            while re.search("I am not a robot", text_, re.IGNORECASE) != None:
                winsound.PlaySound("Dependencies/warning.wav", winsound.SND_FILENAME)
                override = input("Solve the 'I am not a robot' puzzle and confirm by clicking ENTER. Enter 'x' to override. ")
                if override == "x":
                    break
                else:
                    text_ = driver.page_source
                    
    except requests.exceptions.ConnectionError:#My connection was actively refused by server. Happens for CNN and also shows no result via browser.
        driver.get(url)
        text_ = driver.page_source
        while re.search("I am not a robot", text_, re.IGNORECASE) or re.search("recaptcha", text_, re.IGNORECASE) != None:
            winsound.PlaySound("Dependencies/warning.wav", winsound.SND_FILENAME)
            override = input("Solve the 'I am not a robot' puzzle and confirm by clicking ENTER. Enter 'x' to override. ")
            if override == "x":
                break
            else:
                text_ = driver.page_source

    soup = BeautifulSoup(text_)
    dataHist = BeautifulSoup(str(soup.findAll("div", {"class": "cmc-table__table-wrapper-outer"})))
    dataHist = BeautifulSoup(str(dataHist.findAll("table")[2])).prettify()
    dataHist = read_html(dataHist, index_col = 0)[0]
    dataHist.index = to_datetime(dataHist.index)
    
    dataHist.to_csv("Output/Prices/" + coin + ".csv", sep = ";")    


#Import Stocktwits Tweets------------------------------------------------------

#For debugging
#driver = webdriver.Edge("msedgedriver.exe")

def unpackSymbol(alist, coinSymbols):
    symbols = ""
    for i in alist:
        symbols += i["symbol"] + ", "
    return(symbols[:-2])

def filterRep(booleans, alist):
    "Filters list by list of booleans and returns list"
    retList = []
    for i in range(len(alist)):
        if booleans[i] == True:
            retList.append(alist[i])
    return(retList)

tweets = {"ID":[], "Symbols":[], "Body":[], "Sentiment":[], "User":[], "Created":[], "Method":[], "CoinIter":[]}

start_ = to_datetime(datetime.today())-timedelta(forXDays)

coinSymbols = ["ETH.X", "ETC.X", "EOS.X", "BCH.X", "BTC.X"]
lastID = -1
hardStop = 0
scrape = 0

execStart = datetime.today()

for coin in tqdm(coinSymbols, desc = "Getting tweets from Stocktwits", leave = False):
    if hardStop == 1:
        break
    completed = 0
    while completed == 0:
        try:
            if scrape == 1:
                raise(KeyError)
            if lastID == -1:
                API = requests.get("https://api.stocktwits.com/api/2/streams/symbol/" + coin + ".json")
                responses = API.json()
            else:
                API = requests.get("https://api.stocktwits.com/api/2/streams/symbol/" + coin + ".json", params = {"max": lastID})
                responses = API.json()
            for i in range(len(responses["messages"])):
                tweety = responses["messages"][i]
                tweets["ID"].append(tweety["id"])
                tweets["Symbols"].append(unpackSymbol(tweety["symbols"], coinSymbols))
                tweets["Body"].append(tweety["body"])
                tweets["User"].append(tweety["user"]["username"])
                tweets["Created"].append(to_datetime(tweety["created_at"]).replace(tzinfo=None))
                try:
                    tweets["Sentiment"].append(tweety["entities"]["sentiment"]["basic"])
                except TypeError:
                    tweets["Sentiment"].append(np.NaN)
                tweets["Method"].append("API")
                tweets["CoinIter"].append(coin)
                
            positionOfLastElement = int(DataFrame(tweets)[DataFrame(tweets)["ID"] == responses["cursor"]["max"]].index[0])
            if tweets["Created"][positionOfLastElement].replace(tzinfo=None) < start_:
                lastID = -1
                completed = 1
            else:
                lastID = DataFrame(tweets)["ID"][positionOfLastElement]
                
        except KeyError:#Happens after too many requests
            while True:
                if scrape == 0:
                    winsound.PlaySound("Dependencies/warning.wav", winsound.SND_FILENAME)
                    confirmation = input("\n_ _ _ ________________________________ _ _ _\n\n*** ERROR 429: Message Limit Exceeded! ***\n_ _ _ ________________________________ _ _ _\n\nLeave the program running and try again in one hour!\n\n\t-> Confirm by typing 'cont' after this message.\n\n\t-> Force the program to stop by typing 'break'.\n\n\t-> Not in compliance with 'robot.txt': Start the webscraping process by typing 'scrape'.\n\nYour choice: ")
                if confirmation == "cont":
                    break
                elif confirmation == "break":
                    print("Aborting program...")
                    completed = 1
                    hardStop = 1
                    break
                elif confirmation == "scrape":
                    scrape = 1
                    driver.get("https://stocktwits.com/symbol/" + coin)
                    
                    #Wait for tweets to appear
                    WebDriverWait(driver, 100).until(ec.presence_of_element_located((By.XPATH, "//*[@class='st_28bQfzV st_1E79qOs st_3TuKxmZ st_3Y6ESwY st_GnnuqFp st_1VMMH6S']")))
                    
                    text_ = driver.page_source
                    while re.search("I am not a robot", text_, re.IGNORECASE) or re.search("recaptcha", text_, re.IGNORECASE) != None:
                        winsound.PlaySound("Dependencies/warning.wav", winsound.SND_FILENAME)
                        override = input("Solve the 'I am not a robot' puzzle and confirm by clicking ENTER. Enter 'x' to override. ")
                        if override == "x":
                            break
                        else:
                            text_ = driver.page_source
                    
                    soup = BeautifulSoup(text_)
                    tweety = soup.findAll("a", {"class": "st_28bQfzV st_1E79qOs st_3TuKxmZ st_3Y6ESwY st_GnnuqFp st_1VMMH6S"})
                    
                    datesOfTweets = []
                    if len(tweety[-1].text) <= 3:
                        datesOfTweets.append(to_datetime(execStart)-timedelta(minutes = int(tweety[-1].text[:-1])))
                    elif 8 >= len(tweety[-1].text) > 3:
                        datesOfTweets.append(to_datetime(str(execStart.month) + "/" + str(execStart.day) + "/" + str(execStart.year)[2:] + ", " + tweety[-1].text, format = "%m/%d/%y, %I:%M %p"))
                    else:
                        datesOfTweets.append(to_datetime(tweety[-1].text, format = "%m/%d/%y, %I:%M %p"))
                    
                    #See if this value is already before the required one, otherwise scroll down the page until a tweet from before our period is reached
                    while datesOfTweets[-1] > start_:
                        
                        try:
                            for _ in range(10):
                                driver.execute_script("window.scrollBy(0,1080)", "")
                            
                        except TimeoutException:
                            pass#anyways will fail the condition and has to restart
                        
                        text_ = driver.page_source
                        soup = BeautifulSoup(text_)
                                                                   #"st_28bQfzV st_1E79qOs st_3TuKxmZ st_3Y6ESwY st_GnnuqFp st_1VMMH6S"
                        tweety = soup.findAll("a", {"class": "st_28bQfzV st_1E79qOs st_3TuKxmZ st_3Y6ESwY st_GnnuqFp st_1VMMH6S"})
                        
                        datesOfTweets = []
                        if len(tweety[-1].text) <= 3:
                            datesOfTweets.append(to_datetime(execStart)-timedelta(minutes = int(tweety[-1].text[:-1])))
                        elif 8 >= len(tweety[-1].text) > 3:
                            datesOfTweets.append(to_datetime(str(execStart.month) + "/" + str(execStart.day) + "/" + str(execStart.year)[2:] + ", " + tweety[-1].text, format = "%m/%d/%y, %I:%M %p"))
                        else:
                            datesOfTweets.append(to_datetime(tweety[-1].text, format = "%m/%d/%y, %I:%M %p"))

                    #Once the valid tweet is there
                    #Code looks inefficient but it's the only feasible way to ensure data quality.
                    #At 8GB of loaded data, Edge becomes very unresponsive and information gets loaded later.
                    #This leads to different lengths in my dictionary.                    
                    datesOfTweets = []
                    for j in tweety:
                        if len(j.text) <= 3:
                            datesOfTweets.append(to_datetime(execStart)-timedelta(minutes = int(j.text[:-1])))
                        elif 8 >= len(j.text) > 3:
                            datesOfTweets.append(to_datetime(str(execStart.month) + "/" + str(execStart.day) + "/" + str(execStart.year)[2:] + ", " + j.text, format = "%m/%d/%y, %I:%M %p"))
                        else:
                            datesOfTweets.append(to_datetime(j.text, format = "%m/%d/%y, %I:%M %p"))
                    
                    allInfo = soup.findAll("div", {"class": "st_24ON8Bp st_1x3QBA7 st_1SZeGna st_3MXcQ5h st_3-tdfjd"})
                    
                    toRemove = repeat(True, len(datesOfTweets))
                    
                    if len(tweets["CoinIter"]) > 0:
                        if tweets["CoinIter"][-1] == coin:
                            lastTweet = tweets["Created"][-1]
                            toRemove = np.where(array(datesOfTweets) >= lastTweet, False, True)
                            #Make sure I dont overwrite retrieved values from API
                            #allInfo = allInfo

                    #datesOfTweets = filterRep(toRemove, datesOfTweets)        
                    tweets["Created"].extend(filterRep(toRemove, datesOfTweets))
                    #Feed Into the database once all formatting is done
                    tweets["ID"].extend(filterRep(toRemove, [i.findAll("a",{"class": "st_28bQfzV st_1E79qOs st_3TuKxmZ st_3Y6ESwY st_GnnuqFp st_1VMMH6S"}, href = True)[0]["href"].split("/")[-1] for i in allInfo]))
                    tweets["Symbols"].extend(filterRep(toRemove, [symbolsToString(i.findAll("a", {"class": "st_1H1PshU st_1SuHTwr"}), ", ")[:-2] for i in allInfo]))
                    tweets["Body"].extend(filterRep(toRemove, [i.findAll("div", {"class": "st_3SL2gug"})[0].text.replace(symbolsToString(i.findAll("a", {"class": "st_1H1PshU st_1SuHTwr"})),"") for i in allInfo]))
                    tweets["User"].extend(filterRep(toRemove, [i.findAll("a",{"class": "st_28bQfzV st_1E79qOs st_3TuKxmZ st_3Y6ESwY st_GnnuqFp st_1VMMH6S"}, href = True)[0]["href"].split("/")[1] for i in allInfo]))
                    SeNtImEnT = [i.findAll("div", {"class": "lib_XwnOHoV lib_3UzYkI9 lib_lPsmyQd lib_2TK8fEo"}) for i in allInfo]
                    for z in range(len(SeNtImEnT)):
                        if len(SeNtImEnT) > 0:
                            for öojadsg in range(len(SeNtImEnT[z])):
                                if SeNtImEnT[z][öojadsg].text in ["Bullish", "Bearish"]:
                                    SeNtImEnT[z] = SeNtImEnT[z][öojadsg].text
                                    break
                                elif öojadsg == len(SeNtImEnT[z])-1:
                                    SeNtImEnT[z] = ""
                        else:
                            SeNtImEnT[z] = ""
                    #Why ever the F* it does get replaced by a list and not an empty string...        
                    for z in range(len(SeNtImEnT)):
                        if SeNtImEnT[z] not in ["Bullish", "Bearish"]:
                            SeNtImEnT[z] = ""
                    
                    tweets["Sentiment"].extend(filterRep(toRemove, SeNtImEnT))
                    tweets["Method"].extend(repeat("Selenium", len(filterRep(toRemove, datesOfTweets))))
                    tweets["CoinIter"].extend(repeat(coin, len(filterRep(toRemove, datesOfTweets))))
                    
                    completed = 1
                    break
                break

#Export to CSV
tweets = DataFrame(tweets)
tweets.to_csv("Output/StockTwits.csv", sep = ";", encoding = "utf-8-sig", index = False)

driver.quit()

input("Press 'ENTER' to exit.")