import kfold_template
import pandas
from sklearn import linear_model

dataset = pandas.read_csv("ols_dataset.csv")
# dataset = pandas.read_csv("logistic_dataset.csv")
target = dataset.iloc[:,2].values
data = dataset.iloc[:,3:10].values

r2_scores = kfold_template.run_kfold(5, data, target, linear_model.LinearRegression() )
print(r2_scores)



