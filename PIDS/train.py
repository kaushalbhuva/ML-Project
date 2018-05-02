##____train.py_______________##
##____Machine learning pproject 1___##
##____Kaushal Bhuva___163074003__##
##____Rajarshi Biswa__163074004__##

import numpy as np                              # import all libraries
import pandas as pd                             
import scipy
import matplotlib.pyplot as plt
import numpy.matlib as npmat
from sklearn import svm
import csv
#from time import time

#__________READING train.csv file________________________________________________________________________
ele = pd.read_csv('train_kb.csv')                # store elements in ele
elematrix= ele.as_matrix()                        # converting elements into matrix form

my_row = elematrix.shape[0]                        # data matrix rows
my_col = elematrix.shape[1]                        # data matrix columns

feature_cnt = my_col - 2                           # feature count= after removing ID and SaleStatus
print("Train_kb.csv Data : ")
print("Datarows",my_row," DataColumn",my_col," Feature count",feature_cnt)


input_mat = npmat.zeros((my_row,feature_cnt))     # initialize input matrix

#________nominal data categorical processing_____________________________________________________________

for i in np.arange(1, feature_cnt+1):                              # as integer data is not iterative, converting into nimeric elements matrix
    if ele.dtypes[i]  == 'object':              
        tmp0 = pd.Categorical(elematrix[:,i]).codes                # converting categorical data into integer codes 
        input_mat[:,i-1] =np.transpose(np.float64(np.matrix(tmp0)))# transpose the matrix [1,1460] to [1460,1] and convert values to type float
    else :
        for j in np.arange(0,my_row):
             if(~(np.isfinite(elematrix[j,i]))):
                elematrix[j,i] = -1
        input_mat[:,i-1] = np.transpose(np.float64(npmat.matrix(elematrix[:,i])))
print("\ninput matrix\n ",input_mat)
print("\ninput matrix shape before deleting missing data",input_mat.shape)
                     
#___________Remove missing feature data_________________________________________________________________

missing_val = npmat.zeros((my_row,1))               # initialize missing values indicator matrix
missing_count = 0                                   # initialize missing count to 0

for i in np.arange(0,feature_cnt):
    for j in np.arange(0,my_row):
        if input_mat[j,i] < 0:                      # if -1 found 
            #print("missing value")
            missing_val[j,0] = 1                    # fill ones in create missing value indicator matrix
            missing_count += 1                      # increment missing count
            input_mat[j,:] = 0                      # fill entire row with zeros

missing_index = npmat.zeros((missing_count,1))      # missing index matrix created to pass it to numpy.delete function to remove rows with zeros
j=0                                                 # initialise j=0 
for i in np.arange(0,my_row):
    if missing_val[i,0] == 1:
        missing_index[j,0]=i
        j+=1
#print("\nmissing value matrix shape",missing_val.shape)
#print("\nmissing count number",missing_count)
#print("\nmissig index matrix shape",missing_index.shape)
#print("\n Missing index matrix\n",missing_index)

input_mat= np.delete(input_mat,missing_index,axis=0)
print("\ninput matrix shape after delete missing data",input_mat.shape)

new_row = input_mat.shape[0]
new_col = input_mat.shape[1]

#_________Features Data normalization_____________________________________________________________________

for k in np.arange(0,new_col):
    maxx = np.max(input_mat[:,k])
    input_mat[:,k] = input_mat[:,k]/maxx  
print("\n Feature Data normalization done")

#_________Output vector___________________________________________________________________________________

output_mat = npmat.zeros((my_row,1))     # initialize output matrix
tmp = pd.Categorical(elematrix[0:my_row,my_col-1]).codes  # mycol-1 refers to 31st column of soldstatus
output_mat[:,0] = np.transpose(np.matrix(tmp.astype(float))) 
print("\nOutput matrix before deleting missing values",output_mat.shape)

output_mat= np.delete(output_mat,missing_index,axis=0)  # axis=0(select rows), delete missing values
print("\nOutput matrix after deleting missing values",output_mat.shape)

"""
#_________Label Data normalization______# not useful______________________________________________________

maxx2 = np.max(output_mat[:,0])
output_mat[:,0] = output_mat[:,0]/maxx2
print("\n Labels Data normalization done")
#_________________________________________________________________________________________________________
"""

#_________Important Features Extraction___________________________________________________________________

ip_mean = npmat.mean(input_mat,0) 				# column means found out
ip_mean_mat = npmat.repmat(ip_mean,new_row,1)		        # use repmat for creating matrix
ip_mean_sub = input_mat - ip_mean_mat			        # subtract mean from columns

ip_mat_cvar = (npmat.transpose(ip_mean_sub)*ip_mean_sub)/my_row # covariance matrix
dg = np.diagonal(ip_mat_cvar)					# variance
dg = np.sqrt(dg)						# std dev
dg = npmat.matrix(dg)					        # matrix form conversion
scaled_cov=np.transpose(dg)*dg					# scaling the covariance to get Coorelation matrix
ip_mat_corel = np.divide(ip_mat_cvar,scaled_cov)		# corelation input matrix


#plt.matshow(ip_mat_corel)					
#plt.show()

corel_thresh = 1.00						# set threshold of corelation
unimp_feat = npmat.zeros((new_row,1));				# initialize unimportant features matrix
unimp_count=0

for i in np.arange(feature_cnt):
     for j in np.arange(i+1,feature_cnt):
          if ip_mat_corel[i,j]>=corel_thresh:
               unimp_count+=1                                   # increase corelation count 
               unimp_feat[j] = 1;				# form corelation matrix


unimp_index = npmat.zeros((unimp_count,1))                      # missing index matrix created to pass it to numpy.delete function to remove cols
j=0                                                             # initialise j=0 
for i in np.arange(0,new_col):
    if unimp_feat[i,0] == 1:
        unimp_index[j,0]=i
        j+=1

print("Unimportant index",unimp_index)
print("Input matrix before deletng unimp features",input_mat.shape)

input_mat= np.delete(input_mat,unimp_index,axis=1)              # axis=1 means columns selected, remove unimportant columns, 3 columns removed

print("Input matrix before deletng unimp features",input_mat.shape)
print(input_mat.shape)

#print("unimp features",unimp_feat)
print(unimp_feat.shape)
#____________________________________________________________________________________________________________


from sklearn.model_selection import train_test_split
Ip_train ,Ip_test , Op_train, Op_test = train_test_split( input_mat, output_mat, test_size=0.4,random_state=0)
# Ip_train= training data ,Ip_test= testing data , Op_train= output labels, Op_test= ground truth
#Op_train = np.array(Op_train).ravel()   # formatting the Op_train as required by .fit function in 1 D array



#________Random forest Classifier using grid search __________________________________________________

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

forest_grid_para ={   "n_estimators": [20], "max_depth": [3, None],
                      "max_features": [1, 3, 10],"min_samples_split": [2, 3, 10],
                      "min_samples_leaf": [1, 3, 10],"bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]
                  }

forest_grid = GridSearchCV(RandomForestClassifier(),param_grid = forest_grid_para, cv = 5)

forest_grid.fit(Ip_train, np.asarray(Op_train).ravel())					

print(forest_grid.best_params_)

meanScore = forest_grid.cv_results_['mean_test_score']
stddevScore = forest_grid.cv_results_['std_test_score']

print()

for mean, stddev, params in zip(meanScore, stddevScore, forest_grid.cv_results_['params']):
     print("%0.3f (+/-%0.03f) for %r" % (mean, stddev * 2, params))

print()

Op_true, Op_pred = Op_test, forest_grid.predict(Ip_test)
print(classification_report(np.asarray(Op_true).ravel(), np.asarray(Op_pred).ravel()))
print()
print("Predicted labels by random forest grid search \n",Op_pred,"\n predicted labels shape", Op_pred.shape)

#________save the Random forest grid model to disk as .pkl______________________________________________

from sklearn.externals import joblib
filename_rfc = 'Random_forest_grid_model.pkl'
joblib.dump(forest_grid, filename_rfc)

#____________________________________________________________________________________


#_________Grid search SVM______________________________________________________________________
# cited from : http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


#svm_grid_para = [ {'C'     : np.power(10,np.arange(5)),
#                   'kernel': ['linear','poly','rbf'],
#                   'degree': np.arange(3,6),
#                   'gamma' : np.power(10.0,-1*np.arange(6)) }] 


#svm_grid_para = [ {'C'     : np.power(10,np.arange(4)),
#                   'kernel': ['linear','poly','rbf'],
#                   'gamma' : np.power(10.0,-1*np.arange(4)) }]

svm_grid_para = [ {'C'     : np.power(10,np.arange(5)),
                   'kernel': ['linear','poly','rbf'],
                   'gamma' : [0.001,0.0001,0.00001] }]

scores = ['precision','recall']			# tp/(tp+fp),tp/(tp+fn)

for score in scores:
    svm_grid = GridSearchCV(SVC(), svm_grid_para, cv = 5, scoring = '%s_macro' % score)

    svm_grid.fit(Ip_train,np.asarray(Op_train).ravel())

    print(svm_grid.best_params_)
    print("SVM Model accuracy :", (svm_grid.best_score_))
    print ("Best score/accuracy index is :",svm_grid.best_index_)

    meanScore = svm_grid.cv_results_['mean_test_score']
    stddevScore = svm_grid.cv_results_['std_test_score']

    print()

    for mean, stddev, params in zip(meanScore, stddevScore, svm_grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, stddev * 2, params))
    print()
    Op_true, Op_pred = Op_test, svm_grid.predict(Ip_test)
    print(classification_report(np.asarray(Op_true).ravel(), np.asarray(Op_pred).ravel()))
    print()
print("Predicted labels by SVM grid search \n",Op_pred,"\n predicted labels shape", Op_pred.shape)

#________save the SVM model to disk as .pkl______________________________________________
from sklearn.externals import joblib
filename_svm = 'SVM_grid_model.pkl'
joblib.dump(svm_grid, filename_svm)
#____________________________________________________________________________________    



#____________Grid Search Nearest Neighbour_______________________________________
# cited from: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid


nn_grid_para = {"metric":['euclidean']}

scores = ['precision','recall']			    # tp/(tp+fp),tp/(tp+fn)

nn_grid = GridSearchCV(NearestCentroid(),param_grid = nn_grid_para,scoring='%s_macro' %scores[0], cv = 5)

nn_grid.fit(Ip_train, np.asarray(Op_train).ravel())						# The optimisation

print(nn_grid.best_params_)

print("Nearest Neighbour Model accuracy :", (nn_grid.best_score_))
print ("NN Best score/accuracy index is :",nn_grid.best_index_)

meanScore = nn_grid.cv_results_['mean_test_score']
stddevScore = nn_grid.cv_results_['std_test_score']

print()

for mean, stddev, params in zip(meanScore, stddevScore, nn_grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, stddev * 2, params))
print()
Op_true, Op_pred = Op_test, nn_grid.predict(Ip_test)
print(classification_report(np.asarray(Op_true).ravel(), np.asarray(Op_pred).ravel()))
print()
print("Predicted labels by NN grid search \n",Op_pred,"\n predicted labels shape", Op_pred.shape)

#________save the nearest neighbour model to disk as .pkl______________________________________________

from sklearn.externals import joblib
filename_nn = 'NN_grid_model.pkl'
joblib.dump(nn_grid, filename_nn)
#____________________________________________________________________________________    
