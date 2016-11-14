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
    
header = np.array(cr.next()) 
data = np.arange(0)
data_header = header[3:]
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

X = np.array(data[:,3:], dtype = float)
X = preprocessing.scale(X)

pca = doPCA(X)
print
print
print pca.components_[0]
print
print
print pca.explained_variance_ratio_.shape
print
print
print pca.explained_variance_ratio_.sum()
print
print
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