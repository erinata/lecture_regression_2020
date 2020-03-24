import pandas
from sklearn import linear_model
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix

def run_kfold(split_number, data, target, machine, use_confusion=0):
	kfold_object = KFold(n_splits = split_number)
	kfold_object.get_n_splits(data)

	results = []
	for training_index, test_index in  kfold_object.split(data):
		# print("Training: ", training_index)
		# print("Test: ", test_index)
		data_training, data_test = data[training_index], data[test_index]
		target_training, target_test = target[training_index], target[test_index]
		machine.fit(data_training, target_training)
		prediction = machine.predict(data_test)
		if use_confusion == 1:
			print(confusion_matrix(target_test, prediction))
		results.append(metrics.r2_score(target_test, prediction))
	return results

if __name__ == '__main__':
	dataset = pandas.read_csv("ols_dataset.csv")
	target = dataset.iloc[:,2].values
	data = dataset.iloc[:,3:10].values
	r2_scores = run_kfold(5, data, target)
	print(r2_scores)






