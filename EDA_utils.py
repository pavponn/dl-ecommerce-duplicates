import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import cv2
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

def plot(num):
	IMG_PATHS = "./data/train_images/"
	sq_num = np.sqrt(num)
	assert sq_num == int(sq_num), "Number of Images must be a perfect Square!"

	sq_num = int(sq_num)
	image_ids = os.listdir(IMG_PATHS)
	random.shuffle(image_ids)
	fig, ax = plt.subplots(nrows=sq_num, ncols=sq_num, figsize=(10, 10))

	for i in range(sq_num):
		for j in range(sq_num):
			idx = i*sq_num + j
			ax[i, j].axis('off')
			img = cv2.imread(IMG_PATHS + '/' + image_ids[idx])
			img = img[:, :, ::-1]
			ax[i, j].imshow(img); ax[i, j].set_title(f'{image_ids[idx]}', fontsize=6.5)

	plt.show()

def plot_from_label(group):
	IMG_PATHS = "./data/train_images/"
	image_list = train_df[train_df['label_group'] == group]
	image_list = image_list['image'].tolist()
	num = len(image_list)

	sq_num = np.sqrt(num)

	sq_num = int(sq_num)
	image_ids = os.listdir(IMG_PATHS)
	random.shuffle(image_ids)
	fig, ax = plt.subplots(nrows=sq_num, ncols=sq_num, figsize=(10, 10))

	path = [os.path.join(IMG_PATHS, x) for x in image_list]

	for i in range(sq_num):
		for j in range(sq_num):
			idx = i*sq_num + j
			ax[i, j].axis('off')
			img = cv2.imread(path[idx])
			img = img[:, :, ::-1]
			ax[i, j].imshow(img)

	plt.show()

def plot_from_title(title):
	IMG_PATHS = "./data/train_images/"
	image_list = train_df[train_df['title'] == title]
	image_list = image_list['image'].tolist()
	num = len(image_list)

	sq_num = np.sqrt(num)
	sq_num = int(sq_num)

	image_ids = os.listdir(IMG_PATHS)
	random.shuffle(image_ids)
	fig, ax = plt.subplots(nrows=sq_num, ncols=sq_num, figsize=(10, 10))
	fig.suptitle(f"Product Name: {title}")
	path = [os.path.join(IMG_PATHS, x) for x in image_list]

	for i in range(sq_num):
		for j in range(sq_num):
			idx = i*sq_num + j
			ax[i, j].axis('off')
			img = cv2.imread(path[idx])
			img = img[:, :, ::-1]
			ax[i, j].imshow(img)

	plt.show()

def get_top_n_words(corpus, n=None):
	vec = CountVectorizer().fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
	words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
	return words_freq[:n]

def get_top_n_bigram(corpus, n=None):
	vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
	words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
	return words_freq[:n]

def get_top_n_trigram(corpus, n=None):
	vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
	words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
	return words_freq[:n]

def plot_bt(x,w,p):
	common_words = x(train_df['title'], 20)
	common_words_df = DataFrame(common_words,columns=['word','freq'])

	plt.figure(figsize=(16, 10))
	sns.barplot(x='freq', y='word', data=common_words_df,palette=p)
	plt.title("Top 20 "+ w , fontsize=16)
	plt.xlabel("Frequency", fontsize=14)
	plt.yticks(fontsize=13)
	plt.xticks(rotation=45, fontsize=13)
	plt.ylabel("");
	return common_words_df