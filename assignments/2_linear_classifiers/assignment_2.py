###############################################################################
# @author           Alan Ramponi (179850)
# @course           Machine Learning
# @date             December 2nd, 2015
# @description      Performance comparison of Naive Bayes, SVM and Random
#                   forest classification algorithms using 10-fold cross
#                   validation with accuracy, f1-score and AUC ROC metrics.
#
# USAGE: python assignment_2.py [n_samples] [n_features], where:
#   * n_samples: the dimension of artificial samples
#   * n_features: the dimension of artificial features
###############################################################################


from sklearn import datasets, cross_validation, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from numpy import argmax, mean
from sys import argv, stdout, exit


def gaussian_naive_bayes(X_train, y_train, X_test):
    """A function that performs a Gaussian Naive Bayes algorithm in order to predict the labels for the test set.

    Args:
        X_train: the data content used for the training step
        y_train: the data classification used for the training step
        X_test: the data content used for the testing step

    Return:
        A prediction of the X_test based on training information
    """
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    return classifier.predict(X_test)

def rbf_support_vector_machine(X_train, y_train, X_test, c):
    """A function that performs a Support Vector Machine algorithm in order to predict the labels for the test set using the best c parameter.

    Args:
        X_train: the data content used for the training step
        y_train: the data classification used for the training step
        X_test: the data content used for the testing step
        C: the best penalty parameter of the error term

    Return:
        A prediction of the X_test based on training information
    """
    classifier = SVC(C=c, kernel='rbf', class_weight='auto')
    classifier.fit(X_train, y_train)

    return classifier.predict(X_test)

def gini_random_forest(X_train, y_train, X_test, n_est):
    """A function that performs a Random Forest algorithm in order to predict the labels for the test set using the best number of estimators.

    Args:
        X_train: the data content used for the training step
        y_train: the data classification used for the training step
        X_test: the data content used for the testing step
        n_est: the best number of trees in the forest

    Return:
        A prediction of the X_test based on training information
    """
    classifier = RandomForestClassifier(n_estimators=n_est, criterion='gini', random_state=None)
    classifier.fit(X_train, y_train)

    return classifier.predict(X_test)

def check_input_correctness():
    """Check the input correctness as regards the syntax.
    """
    if (len(argv) != 3):
        print "\n"
        print ' ' + '*' * 78
        print " USAGE: python %s [n_samples] [n_features]\n" % argv[0]
        print "\tn_samples\t\tthe dimension of artificial samples"
        print "\tn_features\t\tthe dimension of artificial features"
        print ' ' + '*' * 78
        print "\n"
        exit(1)


# Check the input correctness
check_input_correctness()

# Constants declaration
n_samples = int(argv[1])        # e.g. 1000
n_features = int(argv[2])       # e.g. 10
k = 10

# Generate an artificial dataset for binary classification
dataset = datasets.make_classification(n_samples=n_samples, n_features=n_features, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2)

# Perform k-fold cross validation, with k=10
k_fold = cross_validation.KFold(n_samples, n_folds=k, shuffle=True, random_state=None)

# Variable declaration
data_content = dataset[0]
data_classification = dataset[1]
accuracy = []
f1_score = []
auc_roc = []

# Create a list of lists in order to store the performances of each algorithm
for i in range(0,3):
    accuracy.append([])
    f1_score.append([])
    auc_roc.append([])

for train_i, test_i in k_fold:
    X_train, X_test = data_content[train_i], data_content[test_i]
    y_train, y_test = data_classification[train_i], data_classification[test_i]

    # Variable declaration
    train_len = len(X_train)
    inner_k = 5

    print "Computing Naive Bayes algorithm..."

    # Train the GAUSSIAN NAIVE BAYES algorithm
    pred_naive_bayes = gaussian_naive_bayes(X_train, y_train, X_test)

    # Store the GAUSSIAN NAIVE BAYES performance scores in the designed arrays
    accuracy[0].append(metrics.accuracy_score(y_test, pred_naive_bayes))
    f1_score[0].append(metrics.f1_score(y_test, pred_naive_bayes))
    auc_roc[0].append(metrics.roc_auc_score(y_test, pred_naive_bayes))

    print "Results are stored.\n"

###############################################################################

    # Variable declaration
    c_values = [1e-02, 1e-01, 1e00, 1e01, 1e02]
    best_c = None
    inner_f1_svm = []
    inner_f1_score_svm = []

    print "Computing Support Vector Machine algorithm..."
    print "Choosing the best C value for SVM..."

    for c in c_values:
        # Perform inner k-fold cross validation (with k=5) on the original training set in order to choose the C parameter for SVM
        inner_k_fold = cross_validation.KFold(train_len, n_folds=inner_k, shuffle=True, random_state=None)

        for inner_train_i, inner_validation_i in inner_k_fold:
            X_tr, X_val = X_train[inner_train_i], X_train[inner_validation_i]
            y_tr, y_val = y_train[inner_train_i], y_train[inner_validation_i]

            # Get the SVM prediction for the i-th train/validation pair
            prediction_i = rbf_support_vector_machine(X_tr, y_tr, X_val, c)

            # Save the F1-Score of the i-th train/validation pair in a vector
            inner_f1_svm.append(metrics.f1_score(y_val, prediction_i))

        # Compute the average of the F1-Scores obtained
        avg_f1_score_svm = sum(inner_f1_svm)/len(inner_f1_svm)
        inner_f1_score_svm.append(avg_f1_score_svm)

    # Pick the C parameter that gives the best F1-Score
    best_c = c_values[argmax(inner_f1_score_svm)]

    print "The best C value is %s." % best_c

    # Train the SVM algorithm
    pred_svm = rbf_support_vector_machine(X_train, y_train, X_test, best_c)

    # Store the SVM performance scores in the designed arrays
    accuracy[1].append(metrics.accuracy_score(y_test, pred_svm))
    f1_score[1].append(metrics.f1_score(y_test, pred_svm))
    auc_roc[1].append(metrics.roc_auc_score(y_test, pred_svm))

    print "Results are stored.\n"

###############################################################################

    n_estimators = [10, 100, 1000]
    best_n = None
    inner_f1_rf = []
    inner_f1_score_rf = []

    print "Computing Random Forest algorithm..."
    print "Choosing the best number of estimators for Random Forest..."

    for n in n_estimators:
        # Perform inner k-fold cross validation (with k=5) on the original training set in order to choose the best number of trees for the forest
        inner_k_fold = cross_validation.KFold(train_len, n_folds=inner_k, shuffle=True, random_state=None)

        for inner_train_i, inner_validation_i in inner_k_fold:
            X_tr, X_val = X_train[inner_train_i], X_train[inner_validation_i]
            y_tr, y_val = y_train[inner_train_i], y_train[inner_validation_i]

            # Get the Rand Forest prediction for the i-th train/validation pair
            prediction_i = gini_random_forest(X_tr, y_tr, X_val, n)

            # Save the F1-Score of the i-th train/validation pair in a vector
            inner_f1_rf.append(metrics.f1_score(y_val, prediction_i))

        # Compute the average of the F1-Scores obtained
        avg_f1_score_rf = sum(inner_f1_rf)/len(inner_f1_rf)
        inner_f1_score_rf.append(avg_f1_score_rf)

    # Pick the n parameter that gives the best F1-Score
    best_n = n_estimators[argmax(inner_f1_score_rf)]

    print "The best number of estimators is %s." % best_n

    # Train the Random Forest algorithm
    pred_rf = gini_random_forest(X_train, y_train, X_test, best_n)

    # Store the Random Forest performance scores in the designed arrays
    accuracy[2].append(metrics.accuracy_score(y_test, pred_rf))
    f1_score[2].append(metrics.f1_score(y_test, pred_rf))
    auc_roc[2].append(metrics.roc_auc_score(y_test, pred_rf))

    print "Results are stored.\n"

###############################################################################

print "======================="
print "Performance evaluation:"
print "=======================\n"

print "ACCURACY:\n====================================================================="
print "iter\tNaive Bayes\t\tSVM\t\t\tRandom Forest"
for i in range(0,10):
    stdout.write("%s\t%.8f\t\t%.8f\t\t%.8f\n" % ((i+1), accuracy[0][i], accuracy[1][i], accuracy[2][i]))
print "====================================================================="
stdout.write("AVG\t%.8f\t\t%.8f\t\t%.8f" % (mean(accuracy[0]), mean(accuracy[1]), mean(accuracy[2])))
print "\n\n"

print "F1-SCORE:\n====================================================================="
print "iter\tNaive Bayes\t\tSVM\t\t\tRandom Forest"
for i in range(0,10):
    stdout.write("%s\t%.8f\t\t%.8f\t\t%.8f\n" % ((i+1), f1_score[0][i], f1_score[1][i], f1_score[2][i]))
print "====================================================================="
stdout.write("AVG\t%.8f\t\t%.8f\t\t%.8f" % (mean(f1_score[0]), mean(f1_score[1]), mean(f1_score[2])))
print "\n\n"

print "AUC ROC:\n====================================================================="
print "iter\tNaive Bayes\t\tSVM\t\t\tRandom Forest"
for i in range(0,10):
    stdout.write("%s\t%.8f\t\t%.8f\t\t%.8f\n" % ((i+1), auc_roc[0][i], auc_roc[1][i], auc_roc[2][i]))
print "====================================================================="
stdout.write("AVG\t%.8f\t\t%.8f\t\t%.8f" % (mean(auc_roc[0]), mean(auc_roc[1]), mean(auc_roc[2])))
print "\n\n"
