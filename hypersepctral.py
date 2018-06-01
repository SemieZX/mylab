
# coding: utf-8

# ## 载入数据

# In[8]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import spectral
from functools import reduce

input_image = loadmat('H:\data\hyp_data.mat')['hyp_data']
output_image = loadmat('H:\data\X.mat')['X']
print(input_image.shape)
print(output_image.shape)
print(np.unique(output_image))



# ## 统计类元素的个数

# In[9]:


dict_k = {}
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        #if output_image[i][j] in [m for m in range(1,17)]:
        if output_image[i][j] in [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16]:
            if output_image[i][j] not in dict_k:
                dict_k[output_image[i][j]]=0
            dict_k[output_image[i][j]] +=1

print (dict_k)
print (reduce(lambda x,y:x+y,dict_k.values()))


# ## 显示地物分类

# In[12]:


ground_truth = spectral.imshow(classes = output_image.astype(int),figsize=(8,8))


# In[15]:


x_color =np.array([[255,255,255],
     [184,40,99],
     [74,77,145],
     [35,102,193],
     [238,110,105],
     [117,249,76],
     [114,251,253],
     [126,196,59],
     [234,65,247],
     [141,79,77],
     [183,40,99],
     [0,39,245],
     [90,196,111],
     [100,24,56],
     [45,67,89],
     [ 120,56,20],
        ])
ground_truth = spectral.imshow(classes = output_image.astype(int),figsize =(9,9),colors=x_color)


# ##  重构需要的类

# In[18]:


need_lable = np.zeros([output_image.shape[0],output_image.shape[1]])
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if output_image[i][j] != 0:
            need_lable[i][j] = output_image[i][j]
print(need_lable)
new_datawithlabel_list = []
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if need_lable[i][j] != 0:
            c2l = list(input_image[i][j])
            c2l.append(need_lable[i][j])
            new_datawithlabel_list.append(c2l)

new_datawithlabel_array = np.array(new_datawithlabel_list)


# ## 标准化数据并存储

# In[20]:


from sklearn import preprocessing
data_D = preprocessing.StandardScaler().fit_transform(new_datawithlabel_array[:,:-1])
data_L = new_datawithlabel_array[:,-1]

import pandas as pd
new = np.column_stack((data_D,data_L))
new_ = pd.DataFrame(new)
new_.to_csv('H:\data\X.csv',header=False,index=False)


# ## 训练模型并存储模型

# In[22]:


import joblib
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd


#split the data
data = pd.read_csv('H:\data\X.csv',header=None)
data = data.as_matrix()
data_D = data[:,:-1]
data_L = data[:,-1]
data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size=0.5)

#训练
clf = SVC(kernel='rbf',gamma=0.125,C=16)
clf.fit(data_train,label_train)
pred = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test, pred)*100
print (accuracy)

joblib.dump(clf, "x.m")


# ## 模型预测

# In[25]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import spectral

input_image = loadmat('H:\data\hyp_data.mat')['hyp_data']
output_image = loadmat('H:\data\X.mat')['X']

testdata = np.genfromtxt('H:\data\X.csv',delimiter=',')
data_test = testdata[:,: -1]
label_test = testdata[:,-1]

clf = joblib.load('x.m')
predict_label = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test,predict_label)*100

print (accuracy)


new_show = np.zeros((output_image.shape[0],output_image.shape[1]))
k = 0
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if output_image[i][j] != 0 :
            new_show[i][j] = predict_label[k]
            k +=1 

ground_truth = spectral.imshow(classes = output_image.astype(int),figsize =(5,5))
ground_predict = spectral.imshow(classes = new_show.astype(int), figsize =(5,5))

