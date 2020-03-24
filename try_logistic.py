import pandas
from sklearn import linear_model

dataset = pandas.read_csv("logistic_dataset.csv")
# print(dataset)

target = dataset.iloc[:,1].values
# print(target)

data = dataset.iloc[:,3:9].values
# print(data)


machine = linear_model.LogisticRegression()

machine.fit(data, target)

X = [
	[24,55,31,3,0,7],
	[40,50,2,5,1,8],
	[3,95,37,3,1,15],
]

results = machine.predict(X)
print(results)



