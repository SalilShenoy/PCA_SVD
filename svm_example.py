#NAME:  svm_braf_hiv.py

"""
This file contains an example of how to train and
carry out predictions using the support vector
machine (SVM) learning algorithm. The version of
SVM that is used will be from the scikit package of
machine learning algorithms/modules that is called
sklearn. 
"""



import csv, sys, os
import numpy as np
#from sklearn import svm, cross_validation, metrics, preprocessing
#from sklearn.svm import SVC
from sklearn import svm, cross_validation, metrics, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
#from sklearn.metrics import classification_report, precision_score, recall_score



#CONSTANTS:
__INFINITY = float('inf')


###
# Read the training data set file. The code to read the file 
# has not been put into a function. Feel free to extract whataver
# portions of the code to use in functions you write for your project.
# NOTE:  The format of the example file used here is likely not the same 
# format as for the FDD dataset. In any case, you might find parts of the
# code helpful. 


#================== HELP FUNS =====================
# Check if argument is a number. Return 0 if not. 
# Need to revisit this to see if returning 0 if item is not a number is
# the correct thing to do. 
def make_sure_isnumber(n, row_index, col_index, compound, header, nan_locs):
    """
    This function checks if n is a number. If not, it returns zero. If it is, it checks if it
    is greater than infinity. If it is, it returns a value that is a np.nan.  This value must be
    cleaned up later. If and when a  np.nan is returned,  nan_locs is appended with
    the tuple (row, col) location of the np.nan iin the dataset

    PRECONDITIONS:  n is a string, index is the position in the header with the character string
                                        name of the descriptor being checked, header is an array of descriptor
                                        names.
    POSTCONDITIONS:  Either n is returned as a float type object or 0 is returned, or a np.nan is returned
                                          to indicate that a data value needs further cleaning.
                                          If np.nan is returned, then nan_locs will be appended with the tuple (row, col) index of the location in
                                          the data array that eill need to be cleaned later. 
    SIDEEFFECTS:  None
    ERROR CONDITIONS: None
    MODIFICATION HISTORY:
        >   March 14, 2013
            Len Wesley
             Created initial version of function.
        > March 25, 2013
           Len Wesley
           Modified to check for infinity in addition to whether n is a valid number. If not, np.nan is returned
           otherwise a valid float number is returned. 
    """
    try:
        # If number is > infinity then return np.nan that will need to be cleaned after dataset is completely read in.
        if n == np.nan  or  float(n) >= __INFINITY  or  float(n) == np.nan:
            print  "*** Encountered value =", n, " for the compound named ", compound," and descriptor named ", header[col_index]
            nan_locs = nan_locs.append((row_index, col_index))
            return np.nan
        return float(n)  # else return the number as a float. 
    except ValueError:
        return 0.

#================  END OF HELP FUNS ==================================


# Now we start the real work
# Open the data set file.
try:
    cr = csv.reader(open("BRAF_train_moe_class.csv"))
except IOError:
    raise IOError("Problems locating or opening the .csv file named 'BRAF_train_moe_class.csv' were encountered.")

# Save the first row that is a header row and convert it to a numpy array
header = np.array(cr.next()) 

# Read in the rest of data, and convert items after 3rd col from string to number.
# Assume col 0 is the compound name, col 1 is CID=Compound ID, and 
# col 2 contains the class info
data = np.arange(0)  # Create an empty 1D array that will hold the training data.

# Extract column header names starting from 4th column and higher
data_header = header[3:]

# List of (row, col) coordinates of np.nan values to be cleaned later
# nan_locs is a mutable list and is modified by the make_sure_isnumber  function if and when its first argument
# is >= infinity or a nan. If  thsi is the case, nan_locs is appended with the list [row, col] that is the row and column in
# the dataset that will need to be cleaned later. 
nan_locs = []   
row_index = 0

for row in cr:
    data_row = row[3:]
    new_data_row = np.arange(0)

    if len(data_header) == len(data_row): 
        for col_index in range(len(data_header)):
                new_data_row = np.concatenate((new_data_row, [(make_sure_isnumber(data_row[col_index], \
                                row_index, col_index, row[0], data_header, nan_locs))])) 
            
        if len(data) > 0:
            data = np.vstack((data, np.concatenate((row[:3], new_data_row))))
        else:
            data = np.concatenate((row[:3], new_data_row))

# Extract compound names
print
print "Shape of data = ", data.shape
compound_names = data[:,0]

# Extract CID numbers 
cid_numbers = data[:,1]

# Extract class information and make sure they are float/int types 
print "data[:,2] = ", data[:,2]
print
class_info = np.array(map(lambda x: int(float(x)), data[:,2]))
#class_info = np.array([int(x) for x in data[:,2] ])

# Make sure class is either 0 or 1. If so, append it to class_info list else raise error.
for c in class_info:
    if c not in [0,1]:
        raise ValueError("The column named ",header[2], " in example_svm_train.csv has a value not equal to 0 or 1.")

# At this point the data set has been read in and 
#  data contains just the data and header contains the column 
#  titles/names  and  class_info contains the class membership (i.e., 1 or 0)
# for each entery (row) in data.

# Now perform "gridding" to help find the best SVM kernel and parameters.
"""
The following variables specify the kernels that we wish to test for.
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.0, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000, 10000] }, \
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 10000] }, \
                    {'kernel': ['poly'], 'degree': [1, 2, 3], 'coef0': [0.0, 1., 2.], 'C': [1, 10, 100, 1000, 10000] }, \
                    {'kernel': ['sigmoid'], 'degree': [1, 2, 3],  'coef0': [0.0, 1., 2.],  'gamma': [0.0, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000, 10000]}  ]

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.0, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100] },\
                    {'kernel': ['linear'], 'C': [1, 10, 100] }]
"""


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.0], 'C': [1, 10] }]
tuned_parameters = [{'kernel': ['poly', 'rbf'], 'gamma': [0.0], 'C': [1, 10] }]
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.0, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100] },\
#                    {'kernel': ['linear'], 'C': [1, 10, 100] }]
                    

# What types of scores do we wist to optimize for
#scores = [ ('accuracy', 'accuracy'), ('average_precision', 'average_precision'), ('recall', 'recall')]
scores = [ ('accuracy', 'accuracy')]


# Create np arrays of the data and class data sets. 
# Common names are X and y respectively
X = np.array(data[:,3:], dtype = float)
X = preprocessing.scale(X)   #  scale data between [-1, 1]
y = np.array(class_info, dtype = int)

print
print "Starting the gridding process."
print
# find out how many class 0 and class 1 entries we have.
# we need to use the nimimun number for cross validation 
# purposes.
num_class_0 = list(y).count(0)
num_class_1 = list(y).count(1)
cv_size = min(num_class_0, num_class_1)

"""
Now we loop through the list of kernels and parameter setting 
to try and get as close as possible to the best setting to
use for our prediction machine. 
"""
for score_name, score_func in scores:
    print "   Tuning SVM parameters for %s" % score_name
    print 

    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, scoring = score_func)
    clf.fit(X, y)
    #"""
    clf_scores = cross_validation.cross_val_score(clf, X, y, cv = cv_size)
    print
    print "CLF SCORES: ==================================="
    print  score_name, ": %0.2f (+/- %0.2f)" % (clf_scores.mean(), clf_scores.std() * 2)
    print "==============================================="
    print "Best parameters set found on development set:"
    print
    print clf.best_estimator_
    print 
    print
    
"""
Below is an example of how to generate training and test data sets
using sklean's  functions. In this example, the test_size=0.2
parameter extracts a test data set that is 20% of the entire
dataset. You can change the percentage to whatever you like, but
values between 20% and 50% are not unreasonable, depe`nding on
the size of the original data set.
"""
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
print "X_train shape =", X_train.shape, "  y_train shape=", y_train.shape
#print "X_test shape =", X==.shape, "  y_test shape=", y_test.shape
print

"""
The following lines train the SVM using our extracted training dataset and
is parameterized based on the gridding results. Then the trained SVM is
used to carry out predictions on the test data set. The percentage 
of accuracy predictions is printed
"""
clf = svm.SVC(kernel='rbf', C=10, gamma = 0.0, degree = 3.0, coef0 = 0.0).fit(X_train, y_train)
print "clf.get_params(deep=True) =", clf.get_params(deep=True)
print "clf.score(X_test, y_test) = {0}%".format(int((clf.score(X_test, y_test) * 10000))/100.)
print "clf.predict(X_test) = ", clf.predict(X_test)
print "clf.decision_function(X_test) = ", clf.decision_function(X_test)
print "======================="
print "clf.score(X_train, y_train) = {0}%".format(int((clf.score(X_train, y_train) * 10000))/100.)
print "clf.predict(X_train) = ", clf.predict(X_train)
print "clf.decision_function(X_train) = ", clf.decision_function(X_train)
print "======================="
print
print
print "#####################################"
"""
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
"""
print "clf.support_ = ", clf.support_
print
print "len(clf.support_) = ", len(clf.support_)
print
print "clf.support_vectors_ = ", clf.support_vectors_
print
print "len(clf.support_vectors_) = ", len(clf.support_vectors_)
print
print "len(clf.support_vectors_[0]) = ", len(clf.support_vectors_[0])
print 
print "clf.n_support_ = ", clf.n_support_
print
print "clf.dual_coef_ = ", clf.dual_coef_
print
print "clf.dual_coef_.shape = ", clf.dual_coef_.shape
print
print "clf.dual_coef_[0] = ", clf.dual_coef_[0]
print
print "len(clf.dual_coef_[0]) = ", len(clf.dual_coef_[0])
print
print "clf.intercept_ = ", clf.intercept_
print "#####################################"
"""
http://stackoverflow.com/questions/20113206/scikit-learn-svc-decision-function-and-predict
"""



