#Salil Shenoy 
#PCA

#imports
import csv
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA

### CONSTANTS:
__INFINITY = float('inf')

#Helper Function
def make_sure_isnumber(n, row_index, col_index, compound, header, nan_locs):
    try:
        # If number is > infinity then return np.nan that will need to be cleaned after dataset is completely read in.
        if n == np.nan  or  float(n) >= __INFINITY  or  float(n) == np.nan:
            print  "*** Encountered value =", n, " for the compound named ", compound," and descriptor named ", header[col_index]
            nan_locs = nan_locs.append((row_index, col_index))
            return np.nan
        return float(n)  # else return the number as a float. 
    except ValueError:
        return 0.

def read_data(cr, data, data_header):
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
                
    return data
                        
#PCA
def doPCA(data):
    print
    print 'data shape ===> ', data.shape
    print
    print
    number_components = min(data.shape[0], data.shape[1])
    pca = PCA(n_components = number_components)
    #pca = PCA()
    pca.fit(data)
    return pca
    
#1. Read the BRAF train data
try:
    cr = csv.reader(open("BRAF_train_moe_class.csv"))
except IOError:
    raise IOError("Problems locating or opening the .csv file named 'BRAF_train_moe_class.csv' were encountered.")
    
try:
    test = csv.reader(open("BRAF_test_moe_class.csv"))
except IOError:
    raise IOError("Problems locating or opening the .csv file named 'BRAF_test_moe_class.csv' were encountered.")
    
header = np.array(cr.next())
header_test = np.array(test.next())

data = np.arange(0)
data_header = header[3:]

data_test= np.arange(0)
data_test_header = header_test[3:]

nan_locs = []   
row_index = 0

data = read_data(cr, data, data_header)
test_data = read_data(test, data_test, data_test_header)

X = np.array(data[:,3:], dtype = float)
X = preprocessing.scale(X)

X_test = np.array(test_data[:,3:], dtype = float)
X_test = preprocessing.scale(X_test)

percent_covariance_to_account_for = 0.7
component = 0
tempSum = 0.0

pca = doPCA(X)
'''
print
print
print pca.explained_variance_ratio_
print
print
print pca.explained_variance_ratio_.shape
print
print
print pca.explained_variance_ratio_.sum()
print
print
'''

while (True):
    if (tempSum < percent_covariance_to_account_for):
        tempSum = tempSum + pca.explained_variance_ratio_[component]
        component=component+1
        
    if (tempSum + pca.explained_variance_ratio_[component] > percent_covariance_to_account_for):
        component = component - 1
        break
    
print 'Temp Sum below threshold ==> ', tempSum
print 'Number of PCs selected ==>', component

selected_components = pca.components_[:,0:component]
#print selected_components

sum_lf_components = abs(selected_components).mean(axis = 1)
print sum_lf_components
print sum_lf_components.shape

features = sorted(range(len(sum_lf_components)), key=lambda k:sum_lf_components[k])
percent_dimension_reduction = 0.7

features = sorted(features[:int(percent_dimension_reduction * len(features))])
print 'Features Selected ===> '
print
print features

#Data using new shortlisted features according threshold values
X = X[:,features]
X_test = X_test[:,features]

print
print X
print
print X_test
"""
print pca.explained_variance_
print
print
print pca.get_params()
print 
print
print 'transforming PCA'
print 
print 
pca_trans =  pca.fit_transform(data, (X.shape[0], X.shape[1])) #data , shape (n_samples, n_features) so I have 243 samples and 3 features ??
print 
print
print pca_trans
print
print
print pca.get_params()
"""