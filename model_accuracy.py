import numpy as np

def give_model_acc(models, X_train, y_train, X_val, y_val):
    acc = []
    for model in models:  # one loop without pruning
        model.learn(X_train, y_train)
        preds = model.predict(X_val)
        model_accuracy = np.sum(preds == y_val) / len(y_val)
        acc.append(model_accuracy)

    for model in models:  # one loop with pruning
        model.learn(X_train, y_train, prune=True)
        preds = model.predict(X_val)
        model_accuracy = np.sum(preds == y_val) / len(y_val)
        acc.append(model_accuracy)

    return acc