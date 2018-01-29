import pandas as pd
import pickle
import xgboost as xgb
import csv

training_file = "../data/training_data.csv"
test_file = "../data/test_data.csv"

def get_training_set(filename):
	ans = []
	with open(filename, 'rb') as f:
		csv_reader = csv.DictReader(f)
		for row in csv_reader:
			ans.append([int(row['Volume']), int(row['Good Type']), int(row['Business Type']),
				int(row['Start Business ID']), int(row['End Business ID']), float(row['Distance']),
				float(row['Price'])])
	return ans

def get_test_set(filename):
	ans = []
	with open(filename, 'rb') as f:
		csv_reader = csv.DictReader(f)
		for row in csv_reader:
			ans.append([int(row['Volume']), int(row['Good Type']), int(row['Business Type']),
				int(row['Start Business ID']), int(row['End Business ID']), float(row['Distance'])])
	return ans
if __name__=='__main__':
	training_data = get_training_set(training_file)
	with open('../data/training.pk1', 'wb') as f:
		pickle.dump(training_data, f)
	test_data = get_test_set(test_file)
	with open('../data/test.pk1', 'wb') as f:
		pickle.dump(test_data, f)
