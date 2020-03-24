import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import metrics

dataset = pandas.read_csv("ols_dataset.csv")
print(dataset)

target = dataset.iloc[:,2].values
print(target)

data = dataset.iloc[:,3:10].values
print(data)

data_training, data_test, target_training, target_test = train_test_split(data, target, test_size = 0.25, random_state=0)

# print("data training")
# print(data_training)
# print("data_test")
# print(data_test)
# print("target_training")
# print(target_training)
# print("target_test")
# print(target_test)

print(data.shape)
print(target.shape)
print(data_training.shape)
print(data_test.shape)
print(target_training.shape)
print(target_test.shape)


# Testing once

machine = linear_model.LinearRegression()
machine.fit(data_training, target_training)
prediction = machine.predict(data_test)

print(prediction)

plt.scatter(target_test, prediction)
plt.xlabel("Target of Test dataset")
plt.ylabel("Model Prediction")

plt.savefig("scatter_test_prediction.png")

print(metrics.r2_score(target_test,prediction))






