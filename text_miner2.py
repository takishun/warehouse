import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
import pandas as pd
from janome.tokenizer import Tokenizer
from collections import Counter
import datetime as dt
import sys
import codecs
import time
import pickle
import os
import csv
import io

def coding_log():
	"""
	このコードの変更、追加、改善の記録をこの関数で管理する。
	「coding_log.__doc__」と入力することで関数内のコメントが呼び出せる。
	記載内容は「日付：変更・追加した関数名：変更内容」で記録
	-------------------------------------------------------

	-------------------------------------------------------
	"""

def document():
	"""
	このファイルの内容を記載し、ドキュメントとします。
	記載内容の構成は以下になります。
	====================================
	
	====================================
	
	"""

cps = []

class CorpusElement:
	def __init__(self, text = '', tokens = [], pn_scores = []):
		self.text = text
		self.tokens = tokens
		self.pn_scores = pn_scores

def ja_tokenize(text,types):
	ja_tokenizer = Tokenizer()
	res = []
	lines = text.split()
	lines = lines[1:]
	for line in lines:
		malist = ja_tokenizer.tokenize(line)
		for tok in malist:
			ps = tok.part_of_speech.split(",")[0]
			if not ps in [types]: continue
			w = tok.base_form
			if w == "*" or w == "": w = tok.surface
			if w == "" or w == "\n": continue
			res.append(w)
	return res

def dict_list(text):
	ja_tokenizer = Tokenizer()
	lines = text.split()
	diclist = []
	for text in lines:
		tokens = ja_tokenizer.tokenize(text)
		element = CorpusElement(text, tokens)
		cps.append(element)

def readcsvFile():
	try:
		data = [[elm for elm in v] for v in csv.reader(open(sys.argv[1],"r"))]
		return data
	except IOError as e:
		sys.exit("Unable to open file: {}".format(e))

def readFile():
	try:
		file = open(sys.argv[1],"r", encoding = "shift-jis")
		lists = file.read() 
		return lists
	except IOError as e:
		sys.exit("Unable to open file: {}".format(e))

def output(data,filename):
	with open(filename + '.csv', 'w', newline='') as f:
		f.write('コメント,マネポジ値\n')
		for key, value in data.items():
			f.write('{},{}\n'.format(key,value))
	f.close()	

def output2(data,filename,dirname):
	with open(dirname + filename + '.csv', 'w', newline='') as f:
		csvWriter = csv.writer(f, lineterminator='\n')
		csvWriter.writerows(data.most_common())
	f.close()

def count(col,data):
	"""頻出単語をカウントする関数"""
	lists = []
	for i in range(len(data)-1):
		lists.append(data[i+1][col])
	counter = Counter(lists)
	return counter

def csvReader(filename,ench,delim = ','):		
	try:
		with codecs.open(filename, "r", ench, "ignore") as file:
			df = pd.read_csv(file, delimiter= delim)
			return df
	except IOError as e:
		sys.exit("Unable to open file: {}".format(e))

def load_pn_dict():
	dic = {}

	with codecs.open('pn/pn_ja.txt', 'r', 'shift-jis') as f:
		lines = f.readlines()
		for line in lines:
			columns = line.split(':')
			dic[columns[0]] = float(columns[3])

	return dic

def get_pn_scores(tokens, pn_dic):
	scores = []
	for surface in [t.surface for t in tokens if t.part_of_speech.split(',')[0] in ['動詞','名詞','形容詞','副詞']]:
		if surface in pn_dic:
			scores.append(pn_dic[surface])
	return scores

if __name__ == "__main__":
	"""
	処理
	"""
	save_name1 = '\\names_'+ sys.argv[1]
	save_name2 = '\\sem_'+ sys.argv[1]
	types1 = '名詞'
	types2 = '形容詞'
	outname = os.path.abspath('.') + '\\output_' + sys.argv[1]
	prop = matplotlib.font_manager.FontProperties(fname=r'C:\Windows\Fonts\meiryo.ttc', size=10)
	data1 = readFile()

	##単語カウント#####
	token = ja_tokenize(data1,types1)
	token2 = ja_tokenize(data1,types2)
	counter = Counter(token)
	counter2 = Counter(token2)
	output2(counter,save_name1,os.path.abspath('.'))
	output2(counter2,save_name2,os.path.abspath('.'))
	################

	##ネガポジ値算出##
	outdic = {}

	# with codecs.open("pn/pn_ja.txt", "r", "shift-jis", "ignore") as file:
	# 	df = pd.read_csv(file, sep = ":",names = ('Word', ' Reading', 'POS', 'PN'))

	dict_list(data1)
	pn_dic = load_pn_dict()
	
	i = 0
	for element in cps:
		element.pn_scores = get_pn_scores(element.tokens, pn_dic)
		if len(cps[i].pn_scores) != 0:
			outdic[io.StringIO(element.text).readline()] = float(sum(element.pn_scores)/float(len(cps[i].pn_scores)))
		i += 1

	output(outdic,outname)
	################