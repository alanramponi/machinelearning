###############################################################################
# @author           Alan Ramponi (179850)
# @course           Machine Learning
# @date             October 25th, 2015
# @description      A random sampler that splits the iris dataset into n_iter
#                   different train and test sets of dims train_len:test_len.
#
# USAGE: python sampler.py [n_iter] [train_len] [test_len], where:
#   * n_iter: number of iterations of splitting
#   * train_len: the absolute number of train samples
#   * test_len: the absolute number of test samples
###############################################################################

from sys import argv, exit
from random import randint
from sklearn import cross_validation, datasets


def create_train_file(name, header, train_X, train_y, train_len, features_len):
    """A function that create a new train set file given the splitting information.

    Args:
        name: a string that represent the name of the file to create
        header: a string that represent the header line to add as first line
        train_X: an already splitted train set (without the output classes)
        train_y: an already splitted train set (with only the output classes)
        train_len: the length of the train set
        features_len: the number of the features of the dataset
    """
    output_file = open(name, "w")

    output_file.write(header)   # write the header line

    line = ""
    for r in range(0, train_len):
        for c in range(0, features_len):
            line = line + str(train_X[r][c]) + ";"
        line = line + id_to_name(train_y[r])
        output_file.write(dot_to_comma(line))
        output_file.write("\n")
        line = ""

    output_file.close()

    print name + " successfully created."

def create_test_file(name, header, test_X, test_y, test_len, features_len):
    """A function that create a new test set file given the splitting information.

    Args:
        name: a string that represent the name of the file to create
        header: a string that represent the header line to add as first line
        test_X: an already splitted test set (without the output classes)
        test_y: an already splitted test set (with only the output classes)
        test_len: the length of the test set
        features_len: the number of the features of the dataset
    """
    output_file = open(name, "w")

    output_file.write(header)   # write the header line

    line = ""
    for r in range(0, test_len):
        for c in range(0, features_len):
            line = line + str(test_X[r][c]) + ";"
        line = line + id_to_name(test_y[r])
        output_file.write(dot_to_comma(line))
        output_file.write("\n")
        line = ""

    output_file.close()

    print name + " successfully created."

def id_to_name(class_id):
    """A function that converts a target class ID to its name.

    Args:
        class_id: an integer that represents a specific target class

    Return:
        A string that represents the target class
    """
    class_name = "Iris-" + iris.target_names[class_id]

    return class_name

def dot_to_comma(line):
    """A function that parse a string to meet the requirements of the Hugin software for the Linux environment.

    Args:
        line: a string that represents an iris sample

    Return:
        A (parsed) string that represents an iris sample
    """
    line = line.replace(".", ",")   # represent floats with ',' instead of '.''
    line = line.replace(" ", "")    # delete white spaces from the line

    return line

def check_input_correctness():
    """Check the input correctness as regards both the syntax and the size of train and test sets w.r.t. the total number of dataset samples.
    """
    if (len(argv) != 4) or (int(argv[2]) + int(argv[3]) != iris.data.shape[0]):
        print "\n"
        print ' ' + '*' * 78
        print " USAGE: python %s [n_iter] [train_len] [test_len]\n" % argv[0]
        print "\tn_iter\t\t\tnumber of iterations of splitting"
        print "\ttrain_len\t\tthe absolute number of train samples"
        print "\ttest_len\t\tthe absolute number of test samples"
        print "\n"
        print " NB: train_len + test_len must be equal to the length of the tot samples (" + str(iris.data.shape[0]) + ")!"
        print ' ' + '*' * 78
        print "\n"
        exit(1)


# Load the iris dataset
iris = datasets.load_iris()

# Check the input correctness
check_input_correctness()

# Variables definition
n_iter = int(argv[1])                   # number of iterations of splitting
train_len = int(argv[2])                # store the number of train samples
test_len = int(argv[3])                 # store the number of test samples
samples_len = iris.data.shape[0]        # store the total number of samples
features_len = iris.data.shape[1]       # store the total number of features
header_line = "sepal_length;sepal_width;petal_length;petal_width;type\n"

# Create the train and test sets .dat files for n_iter times
for i in range(0, n_iter):
    rand_number = randint(0, 100)
    train_name = "train_" + str(i+1) + ".dat"
    test_name = "test_" + str(i+1) + ".dat"

    train_X, test_X, train_y, test_y = cross_validation.train_test_split(
        iris.data, iris.target,
        train_size=train_len, test_size=test_len,
        random_state=rand_number
    )

    create_train_file(                  # create the i-th train set file
        train_name, header_line, train_X, train_y, train_len, features_len
    )

    create_test_file(                   # create the i-th test set file
        test_name, header_line, test_X, test_y, test_len, features_len
    )
