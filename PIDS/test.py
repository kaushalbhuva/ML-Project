##____test.py_____________________##
##____Machine Learning Project 1__##
##____Kaushal Bhuva___163074003___##
##____Rajarshi Biswa___163074004__##

import scipy                                        # import necessary methods
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib as npmat
import pandas as pd
from sklearn import svm
import csv

#Ip = pd.read_csv('test_kb.csv', index_col='Id')
#Op = pd.read_csv('gt.csv',index_col='Id')

#__________READING test_kb.csv file_________________________________________________
ele = pd.read_csv('test_kb2.csv')                 # store elements in ele
elematrix= ele.as_matrix()                       # converting elements into matrix form

#print(elematrix)
my_row = elematrix.shape[0]                        # data matrix rows
my_col = elematrix.shape[1]                        # data matrix columns

feature_cnt = my_col - 1                           # feature count= after removing ID
#print("Data Matrix",elematrix)
print("Datarows",my_row)
print("DataColumn",my_col)

#________nominal data categorical processing_____________________________________________________________
input_mat = npmat.zeros((my_row,feature_cnt))                       # initialize input matrix

for i in np.arange(1, feature_cnt+1):                               # as integer data is not iterative, converting into nimeric elements matrix
    if ele.dtypes[i]  == 'object':              
        tmp0 = pd.Categorical(elematrix[:,i]).codes                 # converting categorical data into integer codes 
        input_mat[:,i-1] =np.transpose(np.float64(np.matrix(tmp0))) # transpose the matrix [1,1460] to [1460,1] and convert values to type float
    else :
        for j in np.arange(0,my_row):
             if(~(np.isfinite(elematrix[j,i]))):
                elematrix[j,i] = -1
        input_mat[:,i-1] = np.transpose(np.float64(npmat.matrix(elematrix[:,i])))
#print("\ninput matrix\n ",input_mat)
print("\ninput matrix shape before deleting missing data",input_mat.shape)
                   
#___________Remove missing feature data____________________________________________

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

#_________Features Data normalization____________________________________________________________

for k in np.arange(0,new_col):
    maxx = np.max(input_mat[:,k])
    input_mat[:,k] = input_mat[:,k]/maxx  
print("\n Data normalization done")

#__________READING gt.csv file_________________________________________________
ele2 = pd.read_csv('gt_kb2.csv')                         # store elements in ele
elematrix2= ele2.as_matrix()                         # converting elements into matrix form
my_row2 = elematrix2.shape[0]                        # data matrix rows
my_col2 = elematrix2.shape[1]                        # data matrix columns
print("GTrows",my_row2)
print("GTColumn",my_col2)
#elematrix2 = ele2['SaleStatus']  # extract last column i.e. output(salary)

#_________Output vector____________________________________________________________

output_mat = npmat.zeros((my_row2,1))     # initialize output matrix
tmp2 = pd.Categorical(elematrix2[0:my_row2,my_col2-1]).codes     # mycol-1 refers to 31st column of soldstatus
output_mat[:,0] = np.transpose(np.float64(np.matrix(tmp2)))
#print("\nOutput matrix before deleting missing values",output_mat)
print("\nOutput matrix shape before deleting missing values",output_mat.shape)

output_mat= np.delete(output_mat,missing_index,axis=0)           # axis=0(select rows), delete missing values
#print("\nOutput matrix after deleting missing values",output_mat)
print("\nOutput matrix shape after deleting missing values",output_mat.shape)

print("GT Matrix\n",elematrix2)
print("GTrows",my_row2)
print("GTColumn",my_col2)


#_________Important Features Extraction___________________________________________________________________
#unimp_index = [6.,26.,29.]                                      # remove unimp features as removed in training 
#input_mat= np.delete(input_mat,unimp_index,axis=1)              # axis=1 means columns selected, remove unimportant columns, 3 columns removed
#__________________________________________________________________________________________

"""
from sklearn.model_selection import train_test_split
Ip_train ,Ip_test , Op_train, Op_test = train_test_split( input_mat, output_mat, test_size=0.4,random_state=0) 
# Ip_train= training data ,Ip_test= testing data , Op_train= output labels, Op_test= ground truth
#Op_train = np.array(Op_train).ravel()   # formatting the Op_train as required by .fit function in 1 D array
"""
print("Final input Matrix\n",input_mat)

#________load Random forest Classifier model from disk__________________________________________________
from sklearn.externals import joblib
filename_rfc = 'Random_forest_grid_model.pkl'
load_model_rfc = joblib.load(filename_rfc)

Op_pred_rfc = load_model_rfc.predict(input_mat)

from sklearn.metrics import classification_report
print(classification_report(np.asarray(output_mat).ravel(), np.asarray(Op_pred_rfc).ravel()))

result_rfc = ('RandomForestAccuracy:{:.3f}'.format(load_model_rfc.score(input_mat,output_mat)))
print(result_rfc)

# __________exporting the prediction to out_rfc.csv______________________________________

Op_pred_rfc = np.asarray(Op_pred_rfc).ravel()						# convert to array type
Op_pred_rfc = Op_pred_rfc.astype(int)							# convert to int type
Op_series_rfc = pd.Series(Op_pred_rfc, dtype="category")				# convert to categories
category_rfc = ['Idle','LadderClimb','StoneThrow','WireCut']					# define categories
Op_series_rfc.cat.categories = ["%s" %category_rfc[k] for k in Op_series_rfc.cat.categories]# place proper categories
my_index = np.arange(elematrix2[0,0],new_row+elematrix2[0,0]) 			        # proper data stored in my_index
df = pd.DataFrame({'Id':my_index,'Sensor Activity':Op_series_rfc})			        # convert to dataframes
df.to_csv('out_rfc.csv', encoding='utf-8', index=False)				        # dump values to out_rfc.csv


#________load SVM Classifier model from disk__________________________________________________

from sklearn.externals import joblib
filename_svm = 'SVM_grid_model.pkl'
load_model_svm = joblib.load(filename_svm)

Op_pred_svm = load_model_svm.predict(input_mat)

from sklearn.metrics import classification_report
print(classification_report(np.asarray(output_mat).ravel(), np.asarray(Op_pred_svm).ravel()))

result_svm = ('SVMAccuracy:{:.3f}'.format(load_model_svm.score(input_mat,output_mat)))
print(result_svm)

# discarding the Nearest neighbour classifier as the accuracy is low
"""
#________load Nearest Neighbour Classifier model from disk__________________________________________________
from sklearn.externals import joblib
filename_nn = 'NN_grid_model.pkl'
load_model_nn = joblib.load(filename_nn)
result_nn = ('NearestNeighbourAccuracy:{:.3f}'.format(load_model_svm.score(Ip_test,Op_test)))
print(result_nn)
"""

# __________exporting the prediction to out_svm.csv__________________________________________________________

Op_pred_svm = np.asarray(Op_pred_svm).ravel()						# convert to array type
Op_pred_svm = Op_pred_svm.astype(int)							# convert to int type
Op_series_svm = pd.Series(Op_pred_svm, dtype="category")				# convert to categories
category_svm = ['Idle','LadderClimb','StoneThrow','WireCut']					# define categories
Op_series_svm.cat.categories = ["%s" %category_svm[m] for m in Op_series_svm.cat.categories]# place proper categories
my_index = np.arange(elematrix2[0,0],new_row+elematrix2[0,0]) 			        # proper data stored in my_index
df = pd.DataFrame({'Id':my_index,'SaleStatus':Op_series_svm})			        # convert to dataframes
df.to_csv('out_svm.csv', encoding='utf-8', index=False)				        # dump values to out_rfc.csv


#__________________space for dumy code snippets_____________________________________________________________
"""
a = input_mat.shape[0]   #row                     
b = input_mat.shape[1]   #col
for i in np.arange(0,b):
    plt.plot(input_mat[:,i])  
plt.show()
"""

"""
print(Ip_train)
print(Op_train)
Op_true,Op_predicted = Op_test,loaded_model.predict(Ip_test)
print(Op_true)
print(Op_true.shape)
print(Op_predicted)
print(Op_predicted.shape)
"""
