import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr
from scipy.signal import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LinearRegression


from random import randint
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch import optim as optim
import torch.autograd as autograd

import Extract_data


#import Classifier as cl
#import universal_HAR

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")




feature_gen_model_path='/scratch/lnw8px/activity_recognition/feature_generator_experiemt/geature_generator.pt'
classifier_path='/scratch/lnw8px/activity_recognition/feature_generator_experiemt/classifier.pt'


#extract data from dataframe
def get_data(f):
    start_acc_time=f['acc_time'].iloc[0]
    end_acc_time=start_acc_time+1/100*len(f)
    acc_time_list=np.array(list(range(int(start_acc_time*1000),int(end_acc_time*1000),10)))/1000
    acc_time_list=acc_time_list[:len(f)]

    start_gyr_time=f['gyr_time'].iloc[0]
    end_gyr_time=start_gyr_time+1/100*len(f)
    gyr_time_list=np.array(list(range(int(start_gyr_time*1000),int(end_gyr_time*1000),10)))/1000
    gyr_time_list=gyr_time_list[:len(f)]

    f['acc_time']=acc_time_list
    f['gyr_time']=gyr_time_list

    data=f[['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z']].values
    return data



#num_samples is the number of samples extracted from each class to make the train set
def divide_data(data_df,num_samples):
    train_df=pd.DataFrame(columns=('label','data'))
    classes=np.unique(data_df['label'].values)
    #extract train data
    for cl in classes:
        train_df=train_df.append(data_df[data_df['label']==cl].sample(num_samples))
    #get the rest of data as test data
    total_idx=data_df.index.values
    train_idx=train_df.index.values
    test_idx=np.setdiff1d(total_idx,train_idx)
    test_df=data_df.loc[test_idx]
    return train_df,test_df

sr=50
segment_len=10*sr
def crop_data(data):
    pad_len=segment_len-data.shape[1]
    if(pad_len>=0):
        data=np.pad(data,pad_width=((0,0),(0,pad_len)))
        return data
    start=np.random.randint(0,(data.shape[1]-segment_len))
    end=start+segment_len
    data=data[:,start:end]
    return data

def get_feature_batch(df,bs=10):
    data=df.sample(n=bs)
    #d=data['data'].apply(np.transpose)
    #data['data']=d
    cropped=data['data'].apply(crop_data)
    cropped=np.array([item for item in cropped])
    cropped=torch.from_numpy(cropped)
    cropped=cropped.float()
    cropped=cropped.to(device)

    labels=data['remapped'].values
    labels=labels.astype(int)
    labels=torch.from_numpy(labels)
    labels=labels.to(device)
    features,out=feature_model(cropped)
    return features,labels

def train_KNN(knn_train_df,knn_test_df,n_neighbors=1):
    train_data=knn_train_df['data']
    train_labels=knn_train_df['label']
    train_data=np.array([val for val in train_data])
    train_labels=np.array([val for val in train_labels])

    test_data=knn_test_df['data']
    test_labels=knn_test_df['label']
    test_data=np.array([val for val in test_data])
    test_labels=np.array([val for val in test_labels])
    
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(train_data, train_labels)

    '''calculate accuracy'''
    pred=neigh.predict(test_data)
    accuracy=np.sum(pred==test_labels)/len(test_labels)
    test_df=pd.DataFrame()
    test_df['label']=test_labels
    test_df['pred']=pred
    total_acc,per_class_acc=get_acc(test_df)
    return accuracy,pred,test_labels,per_class_acc



'''
training the similarity classifier 
'''

def get_acc(df):
    df['iscorrect']=df['label']==df['pred']
    per_class_acc=df[['label','iscorrect']].groupby(['label']).mean()
    iscorrect=df['iscorrect'].values
    acc=np.sum(iscorrect)/len(iscorrect)
    return acc,per_class_acc

def get_euclidian_dist(vec,test_vec):
    return np.sqrt(np.sum(np.square(vec-test_vec)))

def get_closest_class(row,centroids):
    centroids['dist']=centroids['data'].apply(get_euclidian_dist,args=[row])
    min_label=centroids[centroids.dist==centroids.dist.min()]['label'].iloc[0]
    return min_label

def train_similarity_classifier(train_df,test_df):
    #calc centroids
    centroids=train_df.groupby(['label'])['data'].apply(np.mean)
    centroids=pd.DataFrame(centroids)
    centroids['label']=centroids.index
    centroids.reset_index(drop=True, inplace=True)
    
    test_df['pred']=test_df['data'].apply(get_closest_class,args=[centroids])
    total_acc,per_class_acc=get_acc(test_df)
    return total_acc,per_class_acc


'''
train svm
'''

def get_numpy_data(df):
    col=df.columns
    out=[]
    for col_name in col:
        data=np.array([item for item in df[col_name].values])
        out.append(data)
    return out

def train_SVM(train_df,test_df):
    eval_train_data=np.array([item for item in train_df['data'].values])
    eval_train_labels=(train_df['label'].values).astype(int)

    eval_test_data=np.array([item for item in test_df['data'].values])
    eval_test_labels=(test_df['label'].values).astype(int)

    #trian
    clf = svm.SVC()
    clf.fit(eval_train_data, eval_train_labels)

    #test
    results=pd.DataFrame()
    results['label']=eval_test_labels
    results['pred']=clf.predict(eval_test_data)
    total_acc,per_class_acc=get_acc(results)
    
    return total_acc,per_class_acc

'''
Linear regression
'''

def get_numpy_data(df):
    col=df.columns
    out=[]
    for col_name in col:
        data=np.array([item for item in df[col_name].values])
        out.append(data)
    return out
    
def train_regressor(train_df,test_df):
    _,eval_train_data,eval_train_labels=get_numpy_data(train_df)
    _,eval_test_data,eval_test_labels=get_numpy_data(test_df)
    #train
    reg = LinearRegression().fit(eval_train_data, eval_train_labels)
    #test
    results=pd.DataFrame()
    results['label']=eval_test_labels
    pred=reg.predict(eval_test_data)
    pred=pred.astype(int)
    results['pred']=pred
    total_acc,per_class_acc=get_acc(results)
    
    return total_acc,per_class_acc


'''
Get iner and inra cluster distances.
data_df : pandas df
must contain columns data : np arrays of data
                    label : int labels of data
                    
model : from library model.py class ModelWork
'''
def get_cluster_distances(data_df,model,sr,segment_len_s):
    labels=np.unique(data_df['label'])
    model=model.to(torch.device("cpu"))

    #calculate intra-cluster distance
    dists_list=[]
    centroid_list=[]
    intra_sd_list=[]
    for label in labels:
        data=data_df[data_df.label==label]
        data,labels,activity_name,_=Extract_data.get_n_times_cropped_data(data,bs=len(data),n_times=1000,
                                                                       sr=sr,
                                                                       segment_len_s=segment_len_s)
        features=model(data)[0]

        centroid=torch.mean(features,dim=0)
        centroid_list.append(centroid)
        centroid=torch.unsqueeze(centroid,dim=0)
        centroid=torch.repeat_interleave(centroid,repeats=features.shape[0],dim=0)

        dists=torch.sqrt(torch.sum(torch.square(centroid-features),dim=1)).detach().cpu()
        stds=torch.mean(torch.std(features,dim=0)).item()
        intra_sd_list.append(stds)
        dists_list.append(torch.mean(dists))
    intra_cluster_dist=np.mean(dists_list)
    intra_sd=np.std(intra_sd_list)
    #calculat inter-cluster distance
    dist_list=[]
    for i in range(len(centroid_list)):
        for j in range(len(centroid_list)):
            if (i==j):
                continue
            dist=torch.sqrt(torch.sum(torch.square(centroid_list[i]-centroid_list[j]))).detach().cpu().item()
            dist_list.append(dist)
    inter_cluster_dist=np.mean(dist_list)
    
    #inter cluster std
    centroid_list=torch.stack(centroid_list)
    centroid_std=torch.mean(torch.std(centroid_list,dim=0)).item()
    
    return intra_cluster_dist,inter_cluster_dist,intra_sd,centroid_std