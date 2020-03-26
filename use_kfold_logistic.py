import kfold_template
import pandas
from sklearn import linear_model

dataset = pandas.read_csv("logistic_dataset.csv")
target = dataset.iloc[:,2].values
data = dataset.iloc[:,3:9].values

r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(5, data, target, linear_model.LogisticRegression(multi_class = "auto",solver="lbfgs"), 1, 1)

print(r2_scores)
print(accuracy_scores)
for confusion_matrix in confusion_matrices:
	print(confusion_matrix)


machine = linear_model.LogisticRegression(multi_class = "auto",solver="lbfgs")
machine.fit(data, target)

X = [
	[24,55,31,3,0,7],
	[40,50,2,5,1,8],
	[3,95,37,3,1,15],
]

results = machine.predict(X)
print(results)



