from decisiontree_tree import DecisionTree
from model_accuracy import give_model_acc
from decisiontree_data import X, y, X_train, X_val, y_train, y_val
from sklearn.model_selection import train_test_split

tree_entropy = DecisionTree(impurity_measure='entropy')
tree_gini = DecisionTree(impurity_measure='gini')

#The first two indexes of acc are entropy and gini, next two are entropy with pruning and gini with pruning.
#This function will be run with some different random splits to more accurately estimate which model is the best.

acc_0 = give_model_acc([tree_entropy, tree_gini],
                     X_train, y_train, X_val, y_val)


X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1)
acc_1 = give_model_acc([tree_entropy, tree_gini],
                     X_train, y_train, X_val, y_val)

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=2)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=2)
acc_2 = give_model_acc([tree_entropy, tree_gini],
                     X_train, y_train, X_val, y_val)

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=4)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=4)
acc_3 = give_model_acc([tree_entropy, tree_gini],
                     X_train, y_train, X_val, y_val)
acc_3

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=5)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=5)
acc_4 = give_model_acc([tree_entropy, tree_gini],
                     X_train, y_train, X_val, y_val)

#Here the final averages are printed.
avg_entropy = (acc_0[0] + acc_1[0] + acc_2[0] + acc_3[0] + acc_4[0]) / 5
avg_gini = (acc_0[1] + acc_1[1] + acc_2[1] + acc_3[1] + acc_4[1]) / 5
avg_entropy_prune = (acc_0[2] + acc_1[2] + acc_2[2] + acc_3[2] + acc_4[2]) / 5
avg_gini_prune = (acc_0[3] + acc_1[3] + acc_2[3] + acc_3[3] + acc_4[3]) / 5
print(f'Entropy: {avg_entropy}, gini: {avg_gini}, entropy pruned: {avg_entropy_prune}, gini pruned: {avg_gini_prune}')