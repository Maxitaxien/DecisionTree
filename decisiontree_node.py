import numpy as np

class DecisionTreeNode:
    def __init__(self, X_data=None, y_data=None, best_gain=0, parent=None, split_val=None, children=[],
                 best_split_index=None, majority_class_label=None):
        self.best_gain = best_gain
        self.parent = parent
        self.children = children
        self.split_val = split_val
        self.best_split_index = best_split_index
        self.X_data = X_data
        self.y_data = y_data
        self.majority_class_label = majority_class_label

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    def find_entropy(self, y):
        entropy = 0

        unique_values = np.unique(y)

        for val in unique_values:
            prob = np.sum(y == val) / len(y)
            if prob > 0:
                entropy -= prob * np.log2(prob)

        return entropy

    def find_gini(self, y):
        gini = 1

        unique_values = np.unique(y)

        for val in unique_values:
            gini -= (np.sum(y == val) / len(y)) ** 2

        return gini

    def find_gain(self, X, y, col, impurity_measure='gini'):
        split_point = np.mean(X[0:, col])

        left = y[X[0:, col] >= split_point]
        right = y[X[0:, col] < split_point]

        if impurity_measure == 'entropy':
            previous_entropy = self.find_entropy(y)

            cond_entropy_l = self.find_entropy(left)
            cond_entropy_r = self.find_entropy(right)

            gain = previous_entropy - (len(left) / len(y)) * cond_entropy_l - (len(right) / len(y)) * cond_entropy_r

        elif impurity_measure == 'gini':
            previous_gini = self.find_gini(y)

            cond_gini_l = self.find_gini(left)
            cond_gini_r = self.find_gini(right)

            gain = previous_gini - (len(left) / len(y)) * cond_gini_l - (len(right) / len(y)) * cond_gini_r

        return gain

    def learn(self, X, y, impurity_measure, min_gain_threshold=0.0):
        # setting data attributes for the node in case it was not done already
        self.y_data = y
        self.X_data = X

        # first checking initial conditions to see if we need to split
        # these will be relevant when doing recursion
        entropy = self.find_entropy(self.y_data)

        if entropy == 0 or np.all(np.all(self.X_data == self.X_data[0], axis=1)):  # if pure y_data or all rows the same
            self.majority_class_label = np.bincount(
                self.y_data).argmax()  # set value to whichever is most prevalent in y_data

        else:  # if not pure y_data or all rows the same, procceed with splitting
            gain_ls = []

            if impurity_measure == 'entropy':
                for col in range(len(X[0,])):
                    gain = self.find_gain(X, y, col, 'entropy')
                    gain_ls.append(gain)


            elif impurity_measure == 'gini':
                for col in range(len(X[0,])):
                    gain = self.find_gain(X, y, col, 'gini')
                    gain_ls.append(gain)

            self.best_gain = gain_ls[np.argmax(gain_ls)]  # choosing the feature that maximizes the information gain

            if self.best_gain > min_gain_threshold:  # if gain is larger than the threshold, by default zero (stopping negative gain)
                self.best_split_index = np.argmax(gain_ls)  # an integer showing which column is selected for splitting
                self.split_val = np.mean(X[:,
                                         self.best_split_index])  # data points below the mean will go to one node, data points above will go to the other

                left_X = X[X[:, self.best_split_index] >= self.split_val]  # X-values for the left branch
                left_y = y[X[:, self.best_split_index] >= self.split_val]  # y-value for the left branch

                right_X = X[X[:, self.best_split_index] < self.split_val]  # X-values for the right branch
                right_y = y[
                    X[:, self.best_split_index] < self.split_val]  # y-value for the right branch

                # Creating the left node for the split
                node_l = DecisionTreeNode(X_data=left_X, y_data=left_y, parent=self, children=[],
                                          best_gain=0, best_split_index=None,
                                          majority_class_label=np.bincount(self.y_data).argmax())
                self.add_child(node_l)  # Adding the left node to the current node

                # Creating the right node for the split
                node_r = DecisionTreeNode(X_data=right_X, y_data=right_y, children=[], parent=self,
                                          best_gain=0, best_split_index=None,
                                          majority_class_label=np.bincount(self.y_data).argmax())

                self.add_child(node_r)  # Adding the right node to the current node

                # Calling the algorithm recursively on the two nodes
                node_l.learn(left_X, left_y, impurity_measure=impurity_measure)
                node_r.learn(right_X, right_y, impurity_measure=impurity_measure)

    def prune(self, X, y):

        leaves_to_prune = self.bottom_up_search()

        while leaves_to_prune:
            current = leaves_to_prune.pop(0)  # check through parent nodes one at a time,
            # simultaneously removing them from the list until it is empty

            # saving nodes children in case we do not wish to prune after all
            saved_children = current.children

            # calculates previous accuracy for pruning data
            previous_acc = np.sum(self.predict(X) == y) / len(y)

            # setting the children to [] means it will return the majority class label
            # (see the else statement in the predict_one method)
            current.children = []

            # calculates new accuracy with the pruning completed
            pruned_acc = np.sum(self.predict(X) == y) / len(y)

            if pruned_acc >= previous_acc:
                # if pruned accuracy is an improvement or not worse, check parent above as well
                if current.parent not in leaves_to_prune:
                    leaves_to_prune.append(current.parent)
            else:
                # set the children back if accuracy was not improved
                current.children = saved_children

    def bottom_up_search(self):
        leaves_to_prune = []

        # defining a new recursive function (such that leaves_to_prune is not overwritten)
        def search(node):
            if not node.children:
                # if the node has no children, it is a leaf, add it's parent to pruning list
                if node.parent not in leaves_to_prune:
                    leaves_to_prune.append(node.parent)
            else:
                # if the node has children, recursively search its children
                for child in node.children:
                    search(child)

        # search begins from the root node (self)
        search(self)

        return leaves_to_prune

    def predict(self, X):
        preds = np.apply_along_axis(self.predict_one, axis=1, arr=X)
        return preds

    def predict_one(self, row):
        current_node = self
        if len(current_node.children) == 2:  # if the node has two children, move down the appropriate child
            if row[current_node.best_split_index] >= current_node.split_val:
                return current_node.children[0].predict_one(row)
            elif row[current_node.best_split_index] < current_node.split_val:
                return current_node.children[1].predict_one(row)
        else:  # if node has no children, return whichever of 1 and 0 is most prevalent in the nodes y_data
            return current_node.majority_class_label