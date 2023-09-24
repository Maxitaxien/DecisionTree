# DecisionTree
A Decision Tree Classifier created from scratch for a project in INF264 (Introduction to machine learning)

How to run the code:

Running instructions:
For running the validation tests i performed, run decisiontree_validation.py. For running the final evaluation,
run decisiontree_evaluation.py. For running the comparison to sklearn's DecisionTreeClassifier,
run decisiontree_sklearn.py. Note that decisiontree_evaluation.py in particular can take a little time to run, as
many models are trained and pruned for this step.


Explanation of other files:
The code is build up into different files, decisiontree_node.py and decisiontree_tree.py,
containing the two different classes (DecisionTreeNode and DecisionTree). The function give_model_acc is contained in
model_accuracy.py. There is also a file (decisiontree_data) that imports and splits the data using random state 0,
which the other files import to use for the models.

These files do not need to be run, and are imported in the other files.
