import numpy as np
from sklearn.tree import DecisionTreeClassifier
from decisiontree_data import X_train, X_val, X_test, y_train, y_val, y_test

sklearn_gini = DecisionTreeClassifier(criterion='gini', random_state=0)
sklearn_entropy = DecisionTreeClassifier(criterion='entropy', random_state=0)

sklearn_gini.fit(X_train, y_train)
sklearn_entropy.fit(X_train, y_train)

sklearn_gini_preds = sklearn_gini.predict(X_val)
sklearn_entropy_preds = sklearn_entropy.predict(X_val)
sklearn_gini_acc = np.sum(sklearn_gini_preds == y_val) / len(y_val)
sklearn_entropy_acc = np.sum(sklearn_entropy_preds == y_val) / len(y_val)

print(f'''Sklearn gini validation accuracy: {sklearn_gini_acc} \n
Sklearn entropy validation accuracy: {sklearn_entropy_acc} \n
Sklearn entropy test accuracy: {np.sum(sklearn_entropy.predict(X_test) == y_test) / len(y_test)}''')