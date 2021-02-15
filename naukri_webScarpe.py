# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 08:32:13 2020

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r"D:\Bhushan\naukri_data4.csv")

data=data_ml_again

data=data.drop_duplicates()


naukri_skill=data['skills']
naukri_skill = ' '.join(naukri_skill)

# stopwords
from nltk.corpus import stopwords
stop= stopwords.words('english')

#skills
import nltk
p=nltk.word_tokenize(naukri_skill)

filtered_skills = [w for w in p if not w in stop]  
filtered_skills = [] 
  
for w in p: 
    if w not in stop: 
        filtered_skills.append(w) 
  
#bigram 1 for skills
from nltk.util import ngrams
bigrams = ngrams(p,2)
list(bigrams)

#most frequent words in skills
from collections import Counter 
split_it = naukri_skill.split() 
 
# Pass the split_it list to instance of Counter class. 
Counter = Counter(filtered_skills)  
most_occur = Counter.most_common(20)  
print(most_occur)   

skills = pd.DataFrame(Counter.most_common(25),
                             columns=['skills', 'count'])
skills.head()
fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
skills.sort_values(by='count').plot.barh(x='skills',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common skills")

plt.show()

#Plotting bigrams for skills
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
vectorizer = CountVectorizer(ngram_range = (2,2)) 
input=[naukri_skill]
X1 = vectorizer.fit_transform(input)  
features = (vectorizer.get_feature_names()) 

#TFIDF
vectorizer = TfidfVectorizer(ngram_range = (2,2)) 
X2 = vectorizer.fit_transform(input) 
scores = (X2.toarray()) 
print("\n\nScores : \n", scores) 
  
# Getting top ranking features 
sums = X2.sum(axis = 0) 
data1 = [] 
for col, term in enumerate(features): 
    data1.append( (term, sums[0,col] )) 
ranking = pd.DataFrame(data1, columns = ['term','rank']) 
words = (ranking.sort_values('rank', ascending = False)) 
print ("\n\nWords head : \n", words.head(5))
# CSV of skills 
words.to_csv('skills.csv')

#Plotting bigrams for skills
fig, ax = plt.subplots(figsize=(8, 8))
# Plot horizontal bar graph
words.head(10).sort_values(by='rank').plot.barh(x='term',
                      y='rank',
                      ax=ax,
                      color="purple")
ax.set_title("Common skills")
plt.show()
plt.hist(words.term[0:10])

#list of unique cities
city_list = data.locations.unique().tolist()

#Plotting most frequent cities
city=data['locations']
city = ' '.join(city)
city = city.replace(',', '')

from collections import Counter 
city = city.split() 

# Pass the city list to instance of Counter class. 
Counter = Counter(city)  
most_occur_city = Counter.most_common(20)  
print(most_occur_city)   

city = pd.DataFrame(Counter.most_common(25),
                             columns=['city', 'count'])
city.head()
fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
city.sort_values(by='count').plot.barh(x='city',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Cities")
plt.show()
#CSv for Cities
city.to_csv('city.csv')

# Unique roles
roles = data.roles.unique().tolist()#616 unique roles
roles = pd.DataFrame(roles)
roles.to_csv("roles.csv")

#Removing strings from experience column
data['experience'] = data['experience'].str.replace('Yrs','', regex=True)
data['experience'] = data['experience'].str.replace('None','0-0', regex=True)

#Splitting range in two columns
data=data.join(data['experience'].str.split('-', 1, expand=True).rename(columns={0:'min_exp', 1:'max_exp'}))
data['min_exp'] =data['min_exp'].astype('float64')
data['max_exp'] =data['max_exp'].apply(pd.to_numeric)
col = data.loc[: , "min_exp":"max_exp"]
data['avg_exp']=col.mean(axis=1)

exp_counts = data['avg_exp'].value_counts()
exp_counts.plot.bar(x='index', y='avg_exp')

exp_counts.to_csv('avg_exp.csv')

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