#Salil Shenoy 
#PCA

#imports
import csv
import numpy as np
from sklearn import preprocessing, cross_validation, svm
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

#Function to Filter, Clean Data
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
    print ' Orignal Data Shape ' 
    print data.shape
    print
    number_components = min(data.shape[0], data.shape[1])
    pca = PCA(n_components = number_components)
    pca.fit(data)
    return pca
    
#1. Read the BRAF train data
try:
    cr = csv.reader(open("BRAF_train_moe_class.csv"))
except IOError:
    raise IOError("Problems locating or opening the .csv file named 'BRAF_train_moe_class.csv' were encountered.")

#2. Read the BRAF test data    
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
data_test = read_data(test, data_test, data_test_header)

class_info_train = np.array(map(lambda x: int(float(x)), data[:,2]))
# Make sure class is either 0 or 1. If so, append it to class_info list else raise error.
for c in class_info_train:
    if c not in [0,1]:
        raise ValueError("The column named ",header[2], " in example_svm_train.csv has a value not equal to 0 or 1.")        
y_train = np.array(class_info_train, dtype = int)

class_info_test = np.array(map(lambda x: int(float(x)), data_test[:,2]))
# Make sure class is either 0 or 1. If so, append it to class_info list else raise error.
for c in class_info_test:
    if c not in [0,1]:
        raise ValueError("The column named ",header_test[2], " in example_svm_train.csv has a value not equal to 0 or 1.")       
y_test = np.array(class_info_test, dtype = int)

X = np.array(data[:,3:], dtype = float)
X = preprocessing.scale(X)

X_test = np.array(data_test[:,3:], dtype = float)
X_test = preprocessing.scale(X_test)

percent_covariance_to_account_for = 0.7
component = 0
tempSum = 0.0

#3. Feature Reduction after PCA 
pca = doPCA(X)

#4. Select number of components based on the covariance threshhold (0.7)
while (True):
    if (tempSum < percent_covariance_to_account_for):
        tempSum = tempSum + pca.explained_variance_ratio_[component]
        component=component+1
        
    if (tempSum + pca.explained_variance_ratio_[component] > percent_covariance_to_account_for):
        component = component - 1
        break
    
print ' Temp Sum below threshold = ', tempSum
print ' Number of PCs selected   = ', component

#5. Selected Principal Componenets
selected_components = pca.components_[:,0:component]
#print selected_components
sum_lf_components = abs(selected_components).mean(axis = 1)

print ' Loading Factors '
print sum_lf_components
print

print ' Shape of List of Loading Factors '  
print sum_lf_components.shape
print

#6. Sorting the features based on the loading factors
features = sorted(range(len(sum_lf_components)), key=lambda k:sum_lf_components[k])
percent_dimension_reduction = 0.7

#7. Selecting the features based on threshold for dimension (in this case n_samples)
features = sorted(features[:int(percent_dimension_reduction * len(features))])
print 'Features Selected'
print features
print

#Data using new shortlisted features according threshold values
X = X[:,features]
X_test = X_test[:,features]

print 'X with new list of features '
print X
print

print ' X test with new list of features'
print X_test
print

#SVM
num_class_0_train = list(y_train).count(0)
num_class_1_train = list(y_train).count(1)
cv_size_train = min(num_class_0_train, num_class_1_train)

num_class_0_test = list(y_test).count(0)
num_class_1_test = list(y_test).count(1)
cv_size_test = min(num_class_0_test, num_class_1_test)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y_train,random_state=0)
print "X_train shape =", X_train.shape, "  y_train shape=", y_train.shape
print "X_test shape =", X_test.shape, "  y_test shape=", y_test.shape
print
clf = svm.SVC(kernel='rbf', C=1, gamma = 0.0, degree = 3.0, coef0 = 0.0).fit(X_train, y_train)

#print "clf.get_params(deep=True) =", clf.get_params(deep=True)

print "clf.score(X_test, y_test) = {0}%".format(int((clf.score(X_test, y_test) * 10000))/100.)
#print "clf.predict(X_test) = ", clf.predict(X_test)
#print "clf.decision_function(X_test) = ", clf.decision_function(X_test)

#print "======================="

print "clf.score(X_train, y_train) = {0}%".format(int((clf.score(X_train, y_train) * 10000))/100.)
#print "clf.predict(X_train) = ", clf.predict(X_train)
#print "clf.decision_function(X_train) = ", clf.decision_function(X_train)