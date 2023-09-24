import pandas as pd

df = pd.read_csv('wine_dataset.csv')
X = df.drop('type', axis=1)
y = df['type']
X = X.to_numpy() #The DecisionTree algorithm is constructed with numpy-arrays in mind
y = y.to_numpy()

from sklearn.model_selection import train_test_split

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=0)


