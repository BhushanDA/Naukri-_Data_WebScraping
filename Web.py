# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:54:06 2020

@author: Administrator
"""

import csv
import time
from selenium import webdriver
import pandas as pd
from bs4 import BeautifulSoup

driver = webdriver.Chrome(r'C:\Users\Administrator\.wdm\drivers\chromedriver\win32\r\chromedriver.exe')

jobs={"roles":[],
     "companies":[],
     "locations":[],
     "experience":[],
     "salary":[],
     "skills":[],
     "job_desc":[]}

for i in range(50):
    driver.get("https://www.naukri.com/machine-learning-jobs-{}".format(i))
    time.sleep(3)
    lst=driver.find_elements_by_css_selector(".jobTuple.bgWhite.br4.mb-8")
    
    # scrape the data from website
    for job in lst:
        driver.implicitly_wait(10)
        try:
            role=job.find_element_by_css_selector("a.title.fw500.ellipsis").text
        except:
            role='None'
        try:
            company=job.find_element_by_css_selector("a.subTitle.ellipsis.fleft").text
        except:
            company='None'
        try:
            location=job.find_element_by_css_selector(".fleft.grey-text.br2.placeHolderLi.location").text
        except:
            location='None'
        try:
            exp=job.find_element_by_css_selector(".fleft.grey-text.br2.placeHolderLi.experience").text
        except:
            exp='None'
        try:
            skills=job.find_element_by_css_selector(".tags.has-description").text
        except:
            skills='None'
        try:
            salary=job.find_element_by_css_selector(".fleft.grey-text.br2.placeHolderLi.salary").text
        except:
            salary='None'
        try:
            job_desc=job.find_element_by_css_selector(".job-description.fs12.grey-text").text
        except:
            job_desc='None'
            
        jobs["roles"].append(role)
        jobs["companies"].append(company)
        jobs["locations"].append(location)
        jobs["experience"].append(exp)
        jobs["skills"].append(skills)
        jobs["salary"].append(salary)
        jobs["job_desc"].append(job_desc)
        
data_ml_again=pd.DataFrame.from_dict(jobs)

import sweetviz as sv
import matplotlib.pyplot as plt

my_report = sv.analyze(data_ml_again)
my_report.show_html()

data_ml_again.to_csv('naukri_data3.csv')

from wordcloud import WordCloud, STOPWORDS 
 

df1=data_ml_again.iloc[:,5:7]

import re

#data cleaning
def preprocessor(text):
    text = re.sub('[^a-zA-Z]', ' ',text)
    text = re.sub('<[^>]*>', ' ', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', ' ')
    return text

df1['job_desc']= df1["job_desc"].apply(preprocessor)
df1['skills']= df1["skills"].apply(preprocessor)

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer() 

def tokenizer_lemmatizer(text):
    return[lem.lemmatize(word, "v") for word in text.split()]

from nltk.corpus import stopwords
stop= stopwords.words('english')

#creating total corpus of mails
corpus = []

for i in data_ml_again.index.values:
    naukri=[w for w in tokenizer_lemmatizer(df1['job_desc'][i]) if w not in stop]
    
    # lem = WordNetLemmatizer()
    # df['content'] = [lem.lemmatize(word, "v") for word in df['content'] if not word in set(stop)]
    naukri = ' '.join(naukri)
    corpus.append(naukri)
    
# creating new cleaned dataset
new_df=pd.DataFrame(list(zip(corpus)), columns=['job_desc'])
text3 = ' '.join(new_df['job_desc'])
wordcloud = WordCloud().generate(text3)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
df1.index.values

#Plotting skill wordcloud
naukri_skill=df1['skills']
naukri_skill = ' '.join(naukri_skill)
# Create and generate a word cloud image:
wordcloud1 = WordCloud().generate(naukri_skill)

# Display the generated image:
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis("off")
plt.show()
df1.index.values

import nltk
p=nltk.word_tokenize(naukri_skill)
p=print(naukri_skill.split()) 

#bigram 
from nltk.util import ngrams
bigrams = ngrams(p,2)
list(bigrams)

#most frequent words
from collections import Counter 
split_it = naukri_skill.split() 
  
# Pass the split_it list to instance of Counter class. 
Counter = Counter(split_it) 
  
most_occur = Counter.most_common(20) 
  
print(most_occur)   

plt.hist(most_occur[5])
