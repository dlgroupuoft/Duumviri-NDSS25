import code
import sys
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from urllib.parse import urlparse, parse_qs, urlunparse,urlencode
from urllib.parse import urlparse
import os
from sklearn.metrics import roc_auc_score
import compress_pickle
import pickle

from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest
import sklearn
from sklearn import tree
import traceback
from sklearn.metrics import f1_score
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn import feature_selection
import pandas as pd
import matplotlib.pyplot as plt

import utils

from tldextract import extract
import xgboost as xgb
import datetime
import numpy as np
from sklearn import metrics
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import glob


def drop_debug_features(df):
	debug_features = {'debug_to_block', 'debug_is_ad', 'response_debug_tracking_apis', 'debug_blocked_retry', 'debug_blocked_url', 'response_all_api_counts', 'debug_vanilla_retry', 'debug_name', 'debug_is_tracker', 'debug_url', '_all_api_counts', '_debug_tracking_apis', 'by_t_oracle_prob', 'by_f_oracle_prob'}

	df2 = df.drop(columns=[c for c in df.columns if c.startswith('debug') or c in debug_features])

	df2 = df2.apply(pd.to_numeric)
	df2 = df2.fillna(0)
	return df2


def evaluate_on_thr (all_data, label, to_block_col_name='debug_to_block', by_us_func=1):
	ret = {}
	thr_range = np.linspace(0.01, 1, 99, endpoint=False)

	for t_thr in thr_range:
		for f_thr in thr_range:

			by_us = (all_data['by_t_oracle_prob'] >= t_thr) & (all_data['by_f_oracle_prob'] < f_thr)
	
			thr_s = '[t-thr:%.2f f_thr:%.2f] accu:%.6f' % (t_thr, f_thr, sklearn.metrics.accuracy_score(label, by_us))
			if t_thr * 100 % 10 == 0 and f_thr * 100 % 10 == 0:
				print(thr_s)
		
			report = metrics.classification_report(label, by_us, output_dict=True, digits=10, zero_division=0)
			ret[(t_thr, f_thr)] = report
	return ret		


def _evaluate_on_request(all_data, t_oracle, f_oracle):
	


	t_cols = t_oracle.feature_names_in_
	f_cols = f_oracle.feature_names_in_

	all_data['by_t_oracle_prob'] = t_oracle.predict_proba(drop_debug_features(all_data)[t_cols])[:, 1]
	all_data['by_f_oracle_prob'] = f_oracle.predict_proba(drop_debug_features(all_data)[f_cols])[:, 1]

	label = all_data['debug_is_ad'] | all_data['debug_is_tracker']
	ret = evaluate_on_thr(all_data, label)

	return ret

def dump(data, path, compressed=False):
	if compressed:
		with open(path, 'wb') as f: 
			compress_pickle.dump(data, f, compression='bz2', set_default_extension=False)
	else:
		pickle.dump(data, open(path, 'wb'))
		# with open(path, 'w') as outf:
		# 	outf.write(data)

def load(path, overwrite=False, compressed=None):
	if compressed is None:
		compressed = path.endswith('pickle')
		
	if compressed:
		try:
			with open(path, 'rb') as f:
				return compress_pickle.load(f, compression='bz2')
		except (OSError, EOFError, ModuleNotFoundError):
			# code.interact('load', local=dict(locals(), **globals()))
	
			with open(path, 'rb') as f: # handle old uncompressed format
				data = pickle.load(open(path, 'rb'))
				
			if overwrite:
				dump(data, path)
			return data
	else:
		try:
			with open(path, 'r') as inf:
				return inf.read()
		except:
			return load(path, overwrite=False, compressed=True)
		
def find_highest_accu(ret):
	h_accu = 0
	h_k = None
	for k, v in ret.items():
		accu = v['accuracy']
		if accu > h_accu:
			h_accu = accu
			h_k = k
	return h_k, h_accu

def load_model(model_path=''):
	if not model_path:
		model_name = max([int(os.path.basename(model_path))
						  for model_path in glob.glob('ml_data/model/*')])
		model_path = 'ml_data/model/%d' % model_name
	model = load(model_path)
	print("Loading %s model " % (os.path.basename(model_path)))
	return model


def replicate():
	t_oracle = load_model('ml_data/model/xgboost_tracker_oracle-20240204') 
	f_oracle = load_model('ml_data/model/xgboost_functionality_oracle-20240204')
	new_data = 'ml_data/all_data.pickle'	
	
	ret_new = _evaluate_on_request(load(new_data), t_oracle, f_oracle)
	print('=================================')
	print("Duumviri achieves the highest accuracy of %.4f  " % find_highest_accu(ret_new)[1])
	print('=================================')


def load_file(fname, drop_debug_columns = 1):
	try:
		df = pd.read_csv(fname, sep=',')  # cached
	except:
		print("Error loading %s" % fname)
		return None
	df = df.fillna(0)
	if drop_debug_columns:
		df = drop_debug_features(df)
	return df
	

def load_list_fnames(fnames, label, drop_debug=False):
	print("Loading %d files" % len(fnames))
	loaded = list(map(load_file, fnames, [drop_debug] * len(fnames)))
	loaded = [x for x in loaded if x is not None]
	T = pd.concat(loaded, ignore_index=True)
	print("After CSV read %d" % len(T))
	T = utils.combine_retries(T)
	print("After combine %d" % len(T))
	# code.interact('drop_duplications', local=dict(locals(), **globals()))
	return T
	

def all_csvs(path):
	ret = []
	for root, dirs, files in os.walk(path):
		for f in files:
			if f.endswith('.csv'):
				ret.append(os.path.join(root, f))
	return ret

def load_data(data_dir):
	
	all_data = []
	csvs = all_csvs(data_dir)
	all_data = load_list_fnames(csvs, None, False)
	return all_data

def add_cols (df, cols):
	for col in cols:
		if col not in df.columns:
			df[col] = 0
	return df

def eval(dir):
	t_oracle = load_model('ml_data/model/xgboost_tracker_oracle-20240204') 
	f_oracle = load_model('ml_data/model/xgboost_functionality_oracle-20240204')
	data = load_data(dir)
	
	t_cols = t_oracle.feature_names_in_
	f_cols = f_oracle.feature_names_in_
	data = add_cols(data, t_cols)
	data = add_cols(data, f_cols)

	data['by_t_oracle_prob'] = t_oracle.predict_proba(drop_debug_features(data)[t_cols])[:, 1]
	data['by_f_oracle_prob'] = f_oracle.predict_proba(drop_debug_features(data)[f_cols])[:, 1]
	t_thr, f_thr = 0.5, 0.5
	data['by_us'] = (data['by_t_oracle_prob'] >= t_thr) & (data['by_f_oracle_prob'] < f_thr)
	
	for idx, row in data.iterrows():
		print(row['debug_to_block'], row['by_us'])


if __name__ == '__main__':
	if sys.argv[1] == 'replicate':
		replicate()
	elif sys.argv[1] == 'eval':
		eval(sys.argv[2])