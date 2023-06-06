## Metrics

### Precision
This is the number of true positives divided by the sum of true positives and false positives. This metric essentially expresses the proportion of positive identifications (the instances where the model predicted the positive class) that were actually correct. A model with high precision is more cautious about making positive predictions, and makes them more accurately.

### Recall:
Also known as sensitivity or true positive rate, this is the number of true positives divided by the sum of true positives and false negatives. This metric expresses the ability of the model to find all the positive instances. High recall means the model correctly identified a large proportion of the positives.

### F1 Score:
The F1 score is the harmonic mean of precision and recall. It tries to balance these two values. This can be more useful than just accuracy, especially if you have an uneven class distribution. A model with a high F1 score is both precise and has a high recall.

### Accuracy:
This is the number of correct predictions made divided by the total number of predictions. In other words, it's the ratio of the sum of true positives and true negatives to the total number of instances. Accuracy can be a useful metric, but it can be misleading if the classes are imbalanced. For example, if 95% of your instances are of the positive class, a model that always predicts positive will be 95% accurate, despite not being a very useful or insightful model.
