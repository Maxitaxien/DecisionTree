from decisiontree_node import DecisionTreeNode
from sklearn.model_selection import train_test_split

class DecisionTree:
    def __init__(self, impurity_measure='entropy', min_gain_threshold=0.0):
        self.impurity_measure = impurity_measure
        self.min_gain_threshold = min_gain_threshold
        self.root = None

    def learn(self, X_train, y_train, prune=False):
        if prune:
            X_train, X_prune, y_train, y_prune = train_test_split(X_train, y_train,
                                                                  test_size=0.2, random_state=42)

            self.root = DecisionTreeNode(X_data=X_train, y_data=y_train, best_gain=0, parent=None, split_val=None,
                                         children=[],
                                         best_split_index=None, majority_class_label=None)

            self.root.learn(X_train, y_train, impurity_measure=self.impurity_measure,
                            min_gain_threshold=self.min_gain_threshold)

            self.prune(X_prune, y_prune)

        else:
            self.root = DecisionTreeNode(X_data=X_train, y_data=y_train, best_gain=0, parent=None, split_val=None,
                                         children=[],
                                         best_split_index=None, majority_class_label=None)
            self.root.learn(X_train, y_train, impurity_measure=self.impurity_measure,
                            min_gain_threshold=self.min_gain_threshold)

    def predict(self, X):
        if self.root is not None:
            return self.root.predict(X)
        else:
            raise ValueError("The decision tree must be trained first.")

    def prune(self, X, y):
        self.root.prune(X, y)
