import numpy as np
from decisiontree_tree import DecisionTree
from decisiontree_data import X_train, X_test, y_train, y_test

final_model = DecisionTree('entropy')
final_model.learn(X_train, y_train, prune=True)
final_preds = final_model.predict(X_test)

print(f'Final model accuracy: {np.sum(final_preds == y_test) / len(y_test)}')