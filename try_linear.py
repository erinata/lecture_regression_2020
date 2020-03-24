import pandas
from sklearn import linear_model

dataset = pandas.read_csv("ols_dataset.csv")
print(dataset)

target = dataset.iloc[:,2].values
print(target)

data = dataset.iloc[:,3:10].values
print(data)

machine = linear_model.LinearRegression()

machine.fit(data, target)

X = [
	[24,55,31,3,0,7,20],
	[40,50,2,5,1,8,20],
	[3,95,37,3,1,15,17],
]

results = machine.predict(X)
print(results)










