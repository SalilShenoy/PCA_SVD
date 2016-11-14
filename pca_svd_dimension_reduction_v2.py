#File Name: pca_svd_dimension_reduction.py
# By: Leonard Wesley
# Date: May 21, 2016
# Python Version(s) 2.7.10 | 64-bit: | Enthought Canopy (default, Oct 21 2015, 17:08:47) [MSC v.1500 64 bit (AMD64)]'

__VERSION = 1.0

### import section
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
import os, csv
import numpy as np
### end of import section 

### CONSTANTS:
__INFINITY = float('inf')
### end of CONSTANTS

### Global variable definition section 
#__DEFAULT_PCA_SVD_COVARIANCE_THRESHOLD = 0.7
#__PCA_SVD_COVARIANCE_THRESHOLD = __DEFAULT_PCA_SVD_COVARIANCE_THRESHOLD
#__PCA_SVD_COVARIANCE_THRESHOLD = __PCA_SVD_COVARIANCE_THRESHOLD
### End Global variable definition section 


### Helper function definition section
def reduce_dimension(X, header, percent_covariance_to_account_for, percent_dimension_reduction): 
    """
    PRECONDITION: X is a 2D numpy array that has beeen normalized between
        -1 and 1. Use the scikit normalize function
                 
                 header is a list of column names of X. 
                 e.g., ["col 1 name", "col 2 name", ..., "col n name"]
                 
                 percent_covariance_to_account_for is the percentage of 
                 the covariance of the data that you wish to account for
                 in the dimension reduction. 
                 
                 percent_dimension_reduction  is the percentage of features
                 listed in a PC that you wish to keep and use when reducing 
                 the dimension.
                 
    POSTCONDITION: A tuple of the reduced data set (X_new) and the corresponding
                 header (header_new) is returned. That is (X_new, header_new) 
    
    SIDE EFFECTS:  None
    
    ERRORS:      None as of VERSION 1 of this code..
    """
    
    # Check if we should use PCA or SVD
    # PCA means Principal Component Analysis, 
    # SVD means Singular Value Decomposition
    X_shape = X.shape # Get dimension of data
    if X_shape[0] == X_shape[1]:  # If number samples equals dimension use PCA else use SVD 
        print
        print "The dimension of the data set is square, and PCA can be used."
        print "However, PCA is not implemented yet, so SVD will be used."
        print
             
    print 'X.shape[1] ===> ', X.shape[1]
    print "====================== Performing SVD ========================="
    num_dimensions = int(percent_dimension_reduction * float(X.shape[1]))
    print ">>>>> num_dimensions=", num_dimensions
    
    # This next section of code simply writes out the header and data
    # that is passed into this function..
    with open('XOUT.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter = ',', quotechar = '|')
        # Now write header
        csvwriter.writerow(header) 
        # Now write the data
        for row in X:
            csvwriter.writerow(row) 
    # End of section to write out original header & data
            
    print "Starting to fit SVD"
    svd = TruncatedSVD(n_components = num_dimensions) 
    svd.fit(X)
    '''
    print
    print "svd.components_: ", svd.components_.shape, " <><><><><>"
    print svd.components_
    print
    print "svd.explained_variance_ratio_:", svd.explained_variance_ratio_.shape ,"<><><><><>"
    print svd.explained_variance_ratio_
    print
    #print "svd.mean_:  <><><><><>"
    #print svd.mean_
    print
    print "svd.explained_variance_:  <><><><><>"
    print svd.explained_variance_
    print
    print "svd.explained_variance_ratio_.sum(): <><><><><>"
    print svd.explained_variance_ratio_.sum()
    print
    print "svd.get_params(): <><><><><>"
    print svd.get_params()
    print 
    print 
    print
    #return True
    '''
    print "Starting to fit SVD Transformed"
    print "==============================="
    svd_trans = TruncatedSVD(n_components = num_dimensions) 
    svd_trans.fit_transform(X,(X.shape[0], 3))
    print
    print "svd_trans = ", svd_trans
    print
    print "svd_trans.components_: ", svd_trans.components_.shape, " <><><><><>"
    print svd_trans.components_
    print
    print
    print svd_trans.components_[0].shape
    print
    print "svd_trans.explained_variance_ratio_:  <><><><><>"
    print svd_trans.explained_variance_ratio_
    print
    print "svd_trans.explained_variance_:  <><><><><>"
    print svd_trans.explained_variance_
    print
    print "svd_trans.explained_variance_ratio_.sum(): <><><><><>"
    print svd_trans.explained_variance_ratio_.sum()
    print
    print "svd_trans.get_params(): <><><><><>"
    print svd_trans.get_params()
    print
    print
    print "Finish building SVD & SVD_TRANSFORMED"
    print
    print "reduce_dimension: RETURNING"
    return True

### End Helper function definition section


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
#compound_names = data[:,0]

# Extract CID numbers 
#cid_numbers = data[:,1]

# Extract class information and make sure they are float/int types 
#print "data[:,2] = ", data[:,2]
#print
#class_info = np.array(map(lambda x: int(float(x)), data[:,2]))
#class_info = np.array([int(x) for x in data[:,2] ])
#class_info = np.array(map(lambda x: int(x), data[:,2]))

#print
#print class_info

# Make sure class is either 0 or 1. If so, append it to class_info list else raise error.
#for c in class_info:
#    if c not in [0,1]:
#        raise ValueError("The column named ",header[2], " in example_svm_train.csv has a value not equal to 0 or 1.")

# At this point the data set has been read in and 
#  data contains just the data and header contains the column 
#  titles/names  and  class_info contains the class membership (i.e., 1 or 0)
# for each entery (row) in data.


# Create np arrays of the data and class data sets. 
# Common names are X and y respectively
n_features = int(raw_input('Enter the number of features you wish to use: '))
n_total_features = int(data.shape[1])
n_features = n_total_features - n_features

X = np.array(data[:,n_features:], dtype = float)

print 'create X after skipping first n_features columns ===>', X.shape[1]

X = preprocessing.scale(X)   #  scale data between [-1, 1]

print 'create X after scaling ===>', X.shape[1]
#print
#print
#print X

#y = np.array(class_info, dtype = int)

percent_covariance_to_account_for = 0.7
percent_dimension_reduction = 0.7


### Check if run or imported
if __name__ == "__main__": # if true then module is executed
    reduce_dimension(X, header, percent_covariance_to_account_for, percent_dimension_reduction)
    #print 'Will be back' 
else: 
    print "pca_svd_dimension_reduction.py: Successfully imported/reloaded\n"