#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 11:12:21 2021

@author: alicehuang
"""

import datetime
import gc
import os
import pandas as pd
import re
import shelve
import time
import datetime
import random


# libraries to crawl websites
from bs4          import BeautifulSoup
from selenium     import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pynput.mouse import Button, Controller
from datetime import timedelta


pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 5)
pd.set_option('display.width',800)
path = '/Users/alicehuang/Desktop/SCHOOL/brandeis/MARKETING ANALYTICS/data files/chromedriver'
driver    = webdriver.Chrome(path)
# os.chdir(path)
os.chdir('/Users/alicehuang/Desktop/SCHOOL/brandeis/MARKETING ANALYTICS/data files')

driver.current_url
driver.close()


# helpful review
# driver.page_source


        
#%%%
driver.get('https://www.sephora.com/product/aha-30-bha-2-peeling-solution-P442563')

# jump to review section
WebDriverWait(driver,10).until(EC.visibility_of_element_located((By.XPATH,"//span[@data-at='number_of_reviews']")))
driver.find_element_by_xpath("//span[@data-at='number_of_reviews']").click()

#scroll until first review appears
mouse = Controller()
position = (576,386)
mouse.position = position 
mouse.scroll (0,-30) 


# Finding all the reviews in the website and bringing them to python
reviews_dict    = []
condition_to_continue = True
while(condition_to_continue):
    reviews               = driver.find_elements_by_xpath("//div[@class='css-13o7eu2 eanm77i0']")
    r = 0
    # Finding all the reviews in the website and bringing them to python
    for r in range(len(reviews)):
        one_review                    = {}
        one_review['scrapping_date']  = datetime.datetime.now()
        one_review['url']             = driver.current_url
        try:
            soup     = BeautifulSoup(reviews[r].get_attribute('innerHTML'))    
        except:
              WebDriverWait(driver,10).until(EC.visibility_of_element_located((By.XPATH,"//div[@class='css-13o7eu2 eanm77i0']")))
              soup    = BeautifulSoup(reviews[r].get_attribute('innerHTML'))        
        try:
            one_review_raw = reviews[r].get_attribute('innerHTML')
        except:
            one_review_raw = ""
        one_review['review_raw'] = one_review_raw
        # pull text from innerHTML 
        try:
            one_review_text = soup.find("div", class_="css-1s11tbv eanm77i0").text
        except:
            one_review_text = ""
        one_review['one_review_text'] = one_review_text
        # pull characteristics
        try:
            one_characteristics = soup.find("span", class_="css-t72irq eanm77i0").text
        except:
            one_characteristics = ""
        one_review['one_characteristics'] = one_characteristics
        #pull stars 
        try:
            one_review_stars = soup.find("div", class_="css-4qxrld")["aria-label"]
        except:
            one_review_stars = ""
        one_review['one_review_stars'] = one_review_stars
        #pull date
        try:
            one_date = soup.find('span', attrs={'data-at':'time_posted'}).text
        except:
            one_date = ""
        one_review['one_date'] = one_date   
        reviews_dict.append(one_review)  
    before = driver.page_source
    driver.find_element_by_xpath('//*[@id="ratings-reviews-container"]/div[2]/ul/li[9]/button').click()
    after = driver.page_source
    if before == after:
        break
    else:
        time.sleep(random.randint(1,3))  
        
    
#%%% Cleaning Data
sephora = pd.DataFrame.from_dict(reviews_dict)
sephora.to_excel('backup-1.xlsx')
# sephora.to_excel('backup1.xlsx')
sephora = pd.read_excel('/Users/alicehuang/Desktop/SCHOOL/brandeis/MARKETING ANALYTICS/data files/backup-1.xlsx',index_col=0).reset_index()
sephora = pd.read_excel('/Users/alicehuang/Desktop/SCHOOL/brandeis/MARKETING ANALYTICS/data files/elephant_backup.xlsx',index_col=0).reset_index()


sephora['review_raw'].map(lambda sephora: BeautifulSoup(sephora).text)
BeautifulSoup(sephora.review_raw.iloc[0]).text

sephora = sephora.dropna(subset = ['one_review_text','one_characteristics'])


#get star rating
sephora['star_rating'] = sephora.one_review_stars.str.extract('([0-5]) (star(s|))')[[0]]
#get skin type
sephora['skin_type']  = sephora.one_characteristics.str.extract('(Combination|Normal|Oily|Dry)')
sephora = sephora.dropna(subset = ['skin_type'])

#drop na


#create a filter 
f = sephora.one_date.str.match('\d+ [d,h] ago')
#extract the rows with d/h ago
tmp = sephora[f].copy().reset_index(drop=True)

tmp1 = tmp.one_date

#change d/h ago to dateformat
i=0
date = []
for i in range(tmp1.shape[0]):
    if re.search(r'(\d+) d ago',str(tmp1.iloc[i])):
        n = str(tmp1.iloc[i]).split(" ")[0]
        d = datetime.date.today() - datetime.timedelta(days = int(n))
        d = d.strftime ('%m/%d/%Y')
        date.append(d)
    elif re.search(r'(\d+) h ago',str(tmp1.iloc[i])):
        d = datetime.date.today()
        d = d.strftime ('%m/%d/%Y')
        date.append(d)    
df = pd.DataFrame(date)
df.columns= ['clean date']
    
#add column with clean date to tmp
tmp2 = pd.concat([tmp,df],axis = 1)

#extract rows without d/h ago
tmp3 = sephora[~f].copy().reset_index(drop=True)
#adjust date format
tmp3['clean date'] = pd.to_datetime(tmp3['one_date']).dt.strftime('%m/%d/%Y')

#concat two df together again
sephora_reviews = pd.concat([tmp2,tmp3],axis = 0)
sephora_reviews = sephora_reviews.drop(['index','one_date','one_review_stars','one_characteristics'], axis = 1)

#fill na for skin_type
# type_list= ['Combination','Dry','Normal','Oily']
# dta['skin_type'] = dta['skin_type'].str.replace(" ",random.choice(type_list))

#to excel
sephora_reviews.to_excel('ordinary-aha.xlsx')
sephora_reviews.to_excel('elephant.xlsx')


