import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn import metrics

dataset = pandas.read_csv("ols_dataset.csv")
print(dataset)

target = dataset.iloc[:,2].values
print(target)

data = dataset.iloc[:,3:10].values
print(data)


kfold_object = KFold(n_splits = 4)
kfold_object.get_n_splits(data)

for training_index, test_index in  kfold_object.split(data):
	print("Training: ", training_index)
	print("Test: ", test_index)
	data_training, data_test = data[training_index], data[test_index]
	target_training, target_test = target[training_index], target[test_index]
	machine = linear_model.LinearRegression()
	machine.fit(data_training, target_training)
	prediction = machine.predict(data_test)
	print(metrics.r2_score(target_test, prediction))

















