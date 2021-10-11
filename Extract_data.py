import pandas as pd
import numpy as np
from random import randint
from scipy.signal import resample
import torch
from os import walk
import math
import random
import csv


def remap(value,classes):
    return np.where(classes==value)[0][0]

def remap_sorted(label,sorted_labels):
    return np.where(sorted_labels==label)[0][0]

def get_remapped_labels(data_df):
    sorted_labels=np.sort(np.unique(data_df['label']))
    return data_df['label'].apply(remap_sorted,args=[sorted_labels])

#interpolate missing data
#e.g. k=data_df['data'].apply(interpolate)
#this apply is in-place
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]
def interpolate(d):
    nans, x= nan_helper(d)
    d[nans]= np.interp(x(nans), x(~nans), d[~nans])
    return d

#standardize data between 0 and 1
def get_standardized(d):
    maxs=np.max(d,axis=1)
    maxs=np.expand_dims(maxs,axis=1)
    maxs=np.repeat(maxs,d.shape[1],axis=1)
    mins=np.min(d,axis=1)
    mins=np.expand_dims(mins,axis=1)
    mins=np.repeat(mins,d.shape[1],axis=1)
    diff=maxs-mins
    diff[diff==0]=0.001
    stand=(d-mins)/(diff)
    return stand

#normalize data (value-mean)/std
def get_normalized(d):
    means=np.mean(d,axis=1)
    means=np.expand_dims(means,axis=1)
    means=np.repeat(means,d.shape[1],axis=1)
    stds=np.std(d,axis=1)
    stds=np.expand_dims(stds,axis=1)
    stds=np.repeat(stds,d.shape[1],axis=1)
    stds[stds==0]=0.001
    stand=(d-means)/(stds)
    return stand

def get_global_standardized(d,max_vals,min_vals):
    max_vals=np.expand_dims(max_vals,axis=1)
    max_vals=np.repeat(max_vals,d.shape[1],axis=1)
    min_vals=np.expand_dims(min_vals,axis=1)
    min_vals=np.repeat(min_vals,d.shape[1],axis=1)
    stand=(d-min_vals)/(max_vals-min_vals)
    return stand

def get_chunks(d,sr,segment_len,step):
    chunks=np.array([d[:,i:i+segment_len] for i in range(0,d.shape[1]-segment_len,step)])
    return chunks
#sr-sample rate in Hz, segment_len and step-in seconds
def get_batch(df,bs,sr,segment_len,step):
    
    data=df.sample(n=bs)
    chunks=data['data'].apply(get_chunks,args=(sr,segment_len,step))

    labels=data['label'].values
    labels=labels.astype(int)
    return chunks,labels,data

#device data into train and test
def divide_data_ratio(data_df,ratio):
    train_data_len=len(data_df)*ratio
    randomized_df=data_df.sample(len(data_df))
    train_df=randomized_df.iloc[0:int(train_data_len)]
    test_df=randomized_df.iloc[int(train_data_len):]
    return train_df,test_df

#num_samples is the number of samples extracted from each class to make the train set
def divide_data_samples(data_df,num_samples):
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

'''
get cropped data suitable for a CNN with fixed input size
'''
def crop_data(data,segment_len):
    pad_len=segment_len-data.shape[1]
    if(pad_len>=0):
        data=np.pad(data,pad_width=((0,0),(0,pad_len)))
        return data
    start=np.random.randint(0,(data.shape[1]-segment_len))
    end=start+segment_len
    data=data[:,start:end]
    return data

#segment_len_s - segment length in seconds
#sr - sample rate 
def get_cropped_batch(df,sr,segment_len_s,bs):
    segment_len=segment_len_s*sr
    data=df.sample(n=bs)
    data['cropped']=data['data'].apply(crop_data,args=[segment_len])
    return data

#get a number of data samples larger than the length of the dataframe
#n_times is how many times of the dataframe is the length of the extracted data
def get_n_times_cropped_data(data_df,bs,n_times,sr=100,segment_len_s=10):
    df=get_cropped_batch(data_df,sr=sr,segment_len_s=segment_len_s,bs=bs)
    for i in range(n_times-1):
        tmp=get_cropped_batch(data_df,sr=sr,segment_len_s=segment_len_s,bs=bs)
        df=df.append(tmp)

    data=df['cropped']
    data=np.array([d for d in data])
    labels=df['label'].values

    labels=torch.from_numpy(labels)

    data=torch.from_numpy(data)
    data=data.float()   
    #activity_vec=np.array([vec for vec in df['activity_vec'].values])
    #activity_vec=torch.from_numpy(activity_vec)
    
    return data,labels,df['activity_name']

#length of all the samples are same as the longest sample
#shorter samples are padded with zero
def get_padded_batch(df,bs):
    selected=df.sample(n=bs)
    selected=selected.copy(deep=True)
    lens=selected['data'].apply(np.shape).values
    max_len=max([l[1] for l in lens])
    padded=selected['data'].apply(lambda x: np.pad(x,((0,0),(0,max_len-x.shape[1])),'constant',constant_values=(0,0)))
    selected['data']=padded
    selected_data=torch.from_numpy(np.array([s for s in selected['data'].values]))
    selected_labels=torch.from_numpy(selected['label'].values)
    activity_names=selected['activity_name'].values
    return selected_data,selected_labels,activity_names



def get_test_acc_loss(model,criterion,data,labels):
    _,pred=model(data)
    predmax=torch.max(pred,1)[1]
    iscorrect=(predmax==labels)
    num_correct=torch.sum(iscorrect).item()
    num_total=iscorrect.shape[0]
    acc=num_correct/num_total
    loss=criterion(pred,labels).item()
    return acc,loss

def get_acc(model,data,labels,pred_index=1):
    ret=model(data)
    pred=ret[pred_index]
    predmax=torch.min(pred,1)[1]
    iscorrect=(predmax==labels)
    num_correct=torch.sum(iscorrect).item()
    num_total=iscorrect.shape[0]
    acc=num_correct/num_total
    return acc

def get_test_acc_by_class(model,data,labels):
    _,pred=model(data)
    predmax=torch.max(pred,1)[1]
    iscorrect=(predmax==labels)
    iscorrect=iscorrect.detach().cpu().numpy()
    labels=labels.detach().cpu().numpy()
    
    df=pd.DataFrame()
    df['iscorrect']=iscorrect
    df['label']=labels
    df=df.groupby(['label']).mean()
    return df
#create sliding windows from the row of data 
def get_windows(row,window_len,overlap):
    data=row['data']
    label=row['label']
    activity_name=row['activity_name']
    participant=row['participant']
    data_list=[]
    pad_len=window_len-data.shape[1]
    if(pad_len>=0):
        data=np.pad(data,pad_width=((0,0),(0,pad_len)))
    if(data.shape[1]<24):
        print(data.shape)
    for i in range(0,(data.shape[1]-window_len+1),(window_len-overlap)):
        data_list.append(data[:,i:(i+window_len)])
    data_array=np.array(data_list)
    label_array=np.array([label]*len(data_list))
    activity_name_array=np.array([activity_name]*len(data_list))
    participant_array=np.array([participant]*len(data_list))
    return data_array,label_array,activity_name_array,participant_array

#get all the data into windows and put them into a daraframe
def get_windowed_df(data_df,window_len,overlap):
    ret=data_df.apply(get_windows,args=[window_len,overlap],axis=1)

    data_array=[item[0] for item in ret]
    label_array=[item[1] for item in ret]
    activity_name_array=[item[2] for item in ret]
    participant_array=[item[3] for item in ret]

    data=[]
    for item in data_array:
        data+=list(item)
    labels=[]
    for item in label_array:
        labels+=list(item)
    names=[]
    for item in activity_name_array:
        names+=list(item)
    participants=[]
    for item in participant_array:
        participants+=list(item)

    windowed_df=pd.DataFrame()
    windowed_df['data']=data
    windowed_df['label']=labels
    windowed_df['activity_name']=names
    windowed_df['participant']=participants
    
    return windowed_df

#extract data from http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
def get_UCI_data(ratio):
    #load data from memory
    df=pd.read_pickle('/scratch/lnw8px/activity_recognition/uci_data/12_Activities/raw_data.pkl')
    classes=np.array(list(range(1,13)))
    df['remapped']=df['activity'].apply(remap,args=[classes])
    df=df.drop(['activity'],axis=1)
    df=df.rename(columns={'remapped':'label'})
    train_len=len(df)*ratio
    randomized_df=df.sample(len(df))
    #standardize data
    #randomized_df['data']=randomized_df['data'].apply(get_standardized)
    
    #for globally standardizing
    #max_vals=np.max(np.array([val for val in randomized_df['data'].apply(np.max,axis=1)]),axis=0)
    #min_vals=np.min(np.array([val for val in randomized_df['data'].apply(np.min,axis=1)]),axis=0)
    #randomized_df['data']=randomized_df['data'].apply(get_global_standardized,args=(max_vals,min_vals))

    train_df=randomized_df.iloc[0:int(train_len)]
    test_df=randomized_df.iloc[int(train_len):]
    return train_df,test_df


#extract data from Ku-HAR dataset
'''
resample data
'''
original_sr=100
required_sr=50
def resample_data(signal,original_sr,required_sr):
    ratio=required_sr/original_sr
    return resample(signal,int(len(signal)*ratio),axis=0)

def get_ku_data():
    ku_df=pd.read_pickle('/scratch/lnw8px/activity_recognition/KU-HAR/extracted_data/Trimmed_raw_data.pkl')
    re=ku_df['data'].apply(resample_data,args=[original_sr,required_sr])
    ku_df['resampled']=re
    classes=np.unique(ku_df['activity'].values)
    print(classes)
    
    #transpose data because handling of earlier batch extraction funcitons
    tr=ku_df['resampled'].apply(np.transpose)
    ku_df['resampled']=tr

    ku_df=ku_df.drop(['data'],axis=1)
    ku_df=ku_df.rename(columns={'resampled':'data','activity':'remapped'})
    return ku_df


'''
extract data from UTWNETE found at https://www.utwente.nl/en/eemcs/ps/research/dataset/
paper Shoaib, Muhammad, Stephan Bosch, Ozlem Durmaz Incel, Hans Scholten, and Paul JM Havinga. "Complex human activity recognition using smartphone and wrist-worn motion sensors." Sensors 16, no. 4 (2016): 426.
'''
def get_UTWNETE_data():
    file='/scratch/lnw8px/activity_recognition/UTWENTE/UT_Data_Complex/smartphoneatwrist.csv'
    data_df=pd.read_csv(file,header=None)
    data_df=data_df.drop([0,1,2,3,10,11,12],axis=1)
    data_df.columns=['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z','label']

    activity_labels=['11111','11112','11113','11114','11115','11116','11117','11118','11119','11120','11121','11122','11123']
    num_chunks=[10,10,10,10,10,10,10,7,7,7,7,6,7]

    data_list=[]
    for i in range(len(activity_labels)):
        d=data_df[data_df['label']==int(activity_labels[i])]
        chunks=np.array_split(d,num_chunks[i])
        for chunk in chunks:
            ar=chunk.drop(['label'],axis=1).values
            ar=np.transpose(ar)
            data_list.append([ar,i])
    data_list=np.array(data_list)
    df=pd.DataFrame(data_list)
    df.columns=['data','label']
    return df

def get_PAMAP2_Data():
    path1='/scratch/lnw8px/activity_recognition/PAMAP2/PAMAP2_Dataset/Protocol/'
    path2='/scratch/lnw8px/activity_recognition/PAMAP2/PAMAP2_Dataset/Optional/'
    
    col=['ts','activity','hr',
     'tmp_hand16',
     'acc16_x_hand','acc16_y_hand','acc16_z_hand',
     'acc6_x_hand','acc6_y_hand','acc6_z_hand',
     'gyr_x_hand','gyr_y_hand','gyr_z_hand',
     'mag_x_hand','mag_y_hand','mag_z_hand',
     'invalid','invalid','invalid','invalid',
     'tmp_chest',
     'acc16_x_chest','acc16_y_chest','acc16_z_chest',
     'acc6_x_chest','acc6_y_chest','acc6_z_chest',
     'gyr_x_chest','gyr_y_chest','gyr_z_chest',
     'mag_x_chest','mag_y_chest','mag_z_chest',
     'invalid','invalid','invalid','invalid',
     'tmp_ankle',
     'acc16_x_ankle','acc16_y_ankle','acc16_z_ankle',
     'acc6_x_ankle','acc6_y_ankle','acc6_z_ankle',
     'gyr_x_ankle','gyr_y_ankle','gyr_z_ankle',
     'mag_x_ankle','mag_y_ankle','mag_z_ankle',
     'invalid','invalid','invalid','invalid']
    
    select_col=['ts','activity',
                'acc16_x_hand','acc16_y_hand','acc16_z_hand',
                'gyr_x_hand','gyr_y_hand','gyr_z_hand',
               'acc16_x_chest','acc16_y_chest','acc16_z_chest',
                'gyr_x_chest','gyr_y_chest','gyr_z_chest',
                'acc16_x_ankle','acc16_y_ankle','acc16_z_ankle',
                'gyr_x_ankle','gyr_y_ankle','gyr_z_ankle',]
    
    '''
    select_col=['ts','activity',
                'acc16_x_hand','acc16_y_hand','acc16_z_hand',
                'gyr_x_hand','gyr_y_hand','gyr_z_hand']
    '''

    _, _, f = next(walk(path1))
    filenames1=[path1+file for file in f if (file.split('.')[-1]=='dat')]

    _, _, f = next(walk(path2))
    filenames2=[path2+file for file in f if (file.split('.')[-1]=='dat')]

    filenames=filenames1+filenames2
    
    data_list,activity_list,participant_list=[],[],[]
    for file in filenames:
        df=pd.read_csv(file,sep=' ',header=None)
        df.columns=col
        df=df[select_col]
        activities=np.unique(df['activity'])
        participant=file.split('/')[-1].split('.')[0][-2:]
        for activity in activities:
            selected=df[df['activity']==activity]
            if((selected['ts'].iloc[-1]-selected['ts'].iloc[0])<2):
                continue
            diffs=np.diff(selected['ts'].values)
            diffs=np.insert(diffs,0,diffs[0])
            selected['diff']=diffs
            selected=selected.reset_index(drop=True)
            break_indices=selected[selected['diff']>0.015].index
            break_indices=list(break_indices)+[selected.index.values[-1]]

            last_index=0
            time=0
            for i in break_indices:
                tmp_df=selected.iloc[last_index:i]
                last_index=i
                if(len(tmp_df)>100):
                    data_list.append(np.array(tmp_df.iloc[:,2:-1].T))
                    activity_list.append(tmp_df['activity'].iloc[0])
                    participant_list.append(participant)
                    time+=tmp_df['ts'].iloc[-1]-tmp_df['ts'].iloc[0]
    df=pd.DataFrame()
    df['data']=data_list
    df['label']=activity_list
    df['participant']=participant_list
    return df

'''
Exract OPP data
'''
class data_reader:
    def __init__(self, dataset, datapath):
        if dataset == 'opportunity':
            self.data, self.id2label = self._read_opportunity(datapath)
            #self.save_data()
        else:
            print('Not supported')
            sys.exit(0)

    def save_data(self):
        f = h5py.File('opportunity.h5')
        for key in self.data:
            f.create_group(key)
            for field in self.data[key]:
                f[key].create_dataset(field, data=self.data[key][field])
        f.close()
        with open('opportunity.h5.classes.json', 'w') as f:
            f.write(json.dumps(self.id2label))

    @property
    def training(self):
        return self.data['training']

    @property
    def test(self):
        return self.data['test']

    @property
    def validation(self):
        return self.data['validation']

    def _read_opportunity(self, datapath):
        files = {
            'training': [
                'S1-ADL1.dat',                'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                'S2-ADL1.dat', 'S2-ADL2.dat',                               'S2-ADL5.dat', 'S2-Drill.dat',
                'S3-ADL1.dat', 'S3-ADL2.dat',                               'S3-ADL5.dat', 'S3-Drill.dat', 
                'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat',
                'S1-ADL2.dat',  'S2-ADL3.dat', 'S2-ADL4.dat', 'S3-ADL3.dat', 'S3-ADL4.dat'

            ],
            'validation': [
            ],
            'test': [
            ]
        }

        label_map = [
            (0,      'Other'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        label2id = {str(x[0]): i for i, x in enumerate(label_map)}
        id2label = [x[1] for x in label_map]

        cols = [
            38, 39, 40, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            64, 65, 66, 67, 68, 69, 70, 71, 72, 77, 78, 79, 80, 81, 82, 83, 84, 85,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
            125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 250]
        cols = [x-1 for x in cols] # labels for 18 activities (including other)

        data = {dataset: self._read_opp_files(datapath, files[dataset], cols, label2id,id2label)
                for dataset in ('training', 'validation', 'test')}

        return data, id2label

    def _read_opp_files(self, datapath, filelist, cols, label2id,id2label):
        data = []
        labels = []
        participants = []
        activity_name = []
        for i, filename in enumerate(filelist):
            participant=int(filename.split('/')[-1].split('-')[0][1:])
            nancnt = 0
            print('reading file %d of %d' % (i+1, len(filelist)))
            with open(datapath.rstrip('/') + '/dataset/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    for ind in cols:
                        elem.append(line[ind])
                    # we can skip lines that contain NaNs, as they occur in blocks at the start
                    # and end of the recordings.
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(label2id[elem[-1]])
                        participants.append(participant)
                        activity_name.append(id2label[label2id[elem[-1]]])
        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)+1,'participants' : np.asarray(participants),'name':np.asarray(activity_name)}


def get_Opportunity_data():
    dr=data_reader('opportunity','/scratch/lnw8px/activity_recognition/OOP/OpportunityUCIDataset/')
    df=pd.DataFrame()
    df['data']=list(dr.training['inputs'])
    df['label']=list(dr.training['targets'])
    df['participant']=list(dr.training['participants'])
    df['activity']=list(dr.training['name'])
    data_list,label_list,activity_name_list,participant_list=[],[],[],[]
    for i, g in df.groupby([(df['label'] != df['label'].shift()).cumsum()]):
        activity=g['label'].iloc[0]
        activity_name=g['activity'].iloc[0]
        participant=g['participant'].iloc[0]
        if((not(g.isnull().values.any())) and (activity!=0)):
            data_list.append(np.transpose(np.array([item for item in g['data'].values])))
            label_list.append(activity)
            participant_list.append(participant)
            activity_name_list.append(activity_name)

    data_df=pd.DataFrame()
    data_df['data']=data_list
    data_df['participant']=participant_list
    data_df['label']=label_list
    data_df['activity_name']=activity_name_list
    
    return data_df


'''
prepare data
'''

'''
prepare UTWNETE data
'''
def prepare_UTWENTE_data():
    data_df=get_UTWNETE_data()
    class_names={
        0:'walk',
        1:'stand',
        2:'jog',
        3:'sit',
        4:'bike',
        5:'upstairs',
        6:'downstairs',
        7:'type',
        8:'write',
        9:'drink',
        10:'talk',
        11:'smoke',
        12:'eat'
    }
    seen_classes=['walk','stand','upstairs','type','drink','talk','smoke','eat']
    unseen_classes=['sit','bike','jog','downstairs','write']

    def get_activity_name(label):
            return class_names[label]
    data_df['activity_name']=data_df['label'].apply(get_activity_name)
    data_df['participant']='1'
    participants=np.array(['1'])
    sr=100
    return data_df,seen_classes,unseen_classes,sr

'''
prepare Opportunity data
'''
def prepare_opportunity_data():
    data_df=get_Opportunity_data()
    #normalize data
    def get_global_standardized(data_df):
        max_vals=np.max(np.array([np.max(item,axis=1) for item in data_df['data']]),axis=0)
        min_vals=np.min(np.array([np.min(item,axis=1) for item in data_df['data']]),axis=0)

        scaled_list=[]
        for d in data_df['data']:
            maxs=np.expand_dims(max_vals,axis=1)
            maxs=np.repeat(maxs,d.shape[1],axis=1)
            mins=np.expand_dims(min_vals,axis=1)
            mins=np.repeat(mins,d.shape[1],axis=1)
            scaled=(d-maxs)/((maxs-mins)+0.001)
            scaled_list.append(scaled)
        return scaled_list
    #data_df['data']=get_global_standardized(data_df)
    participants=np.unique(data_df['participant'])

    seen_classes=["Open Door 2","Close Door 2","Close Fridge","Close Dishwasher","Close Drawer 1",
                 "Close Drawer 2","Close Drawer 3",
                 "Clean Table","Drink from Cup","Toggle Switch"]
    unseen_classes=["Open Door 1","Close Door 1","Open Fridge",
                   "Open Dishwasher","Open Drawer 1","Open Drawer 2","Open Drawer 3"]
    sr=30
    return data_df,seen_classes,unseen_classes,sr

'''
prepare PAMAP2 data
'''
def prepare_PAMAP2_data():
    #load data from PAMAP2 dataset
    data_df=get_PAMAP2_Data()
    #interpolate NAN values
    k=data_df['data'].apply(interpolate)

    class_names={
        0:'other',
        1:'lying',
        2:'sitting',
        3:'standing',
        4:'walking',
        5:'running',
        6:'cycling',
        7:'Nordic walking',
        9:'watching TV',
        10:'computer work',
        11:'car driving',
        12:'ascending stairs',
        13:'descending stairs',
        16:'vacuum cleaning',
        17:'ironing',
        18:'folding laundry',
        19:'house cleaning',
        20:'playing soccer',
        24:'rope jumping'  
    }


    def get_activity_name(label):
            return class_names[label]
    data_df['activity_name']=data_df['label'].apply(get_activity_name)

    data_df=data_df[data_df['activity_name']!='other']
    data_df['participant']=data_df['participant'].apply(np.int)
    data_df['participant']=data_df['participant']-1

    def get_group(i):
        return int(i/3)
    data_df['participant']=data_df['participant'].apply(get_group)

    participants=np.unique(data_df['participant'])
    '''
    class divisions mentioned in paper
    Few-Shot Learning-Based Human Activity Recognition by
    Feng and Duarte
    '''
    seen_classes=['lying','standing','walking','running','ascending stairs','vacuum cleaning','rope jumping']
    unseen_classes=['sitting','cycling','Nordic walking','descending stairs','ironing']
    sr=100
    
    return data_df,seen_classes,unseen_classes,sr


'''
Make class divisions
'''

def get_label_location(label,class_names):
    idx=[i for i,name in enumerate(class_names) if name==label][0]
    return idx

def get_seen_unseen_data(data_df,source_classes,target_classes,source_group,target_group):
    source_df=data_df[data_df['participant'].isin(source_group)]
    target_df=data_df[data_df['participant'].isin(target_group)]

    source_data=source_df[source_df['activity_name'].isin(source_classes)]
    target_data=target_df[target_df['activity_name'].isin(target_classes)]

    source_data['label']=source_data['activity_name'].apply(get_label_location,args=[source_classes])
    target_data['label']=target_data['activity_name'].apply(get_label_location,args=[target_classes])
    
    return source_data,target_data

def get_random_seen_unseen_data(data_df,class_names,num_unseen_classes):
    keys=list(class_names.keys())
    random.shuffle(keys)
    unseen_classes=keys[0:num_unseen_classes]
    seen_classes=keys[num_unseen_classes:]
    seen_data=data_df[data_df['label'].isin(seen_classes)]
    unseen_data=data_df[data_df['label'].isin(unseen_classes)]

    seen_data['label']=get_remapped_labels(seen_data)
    unseen_data['label']=get_remapped_labels(unseen_data)

    return seen_data,unseen_data,unseen_classes

#randomly select data belopnging to num_selected number of classes
def select_random_classes_data(df,num_selected=3):
    classes=df['label'].unique()
    random.shuffle(classes)
    selected_classes=classes[0:num_selected]
    other_classes=classes[num_selected:]
    selected_df=df[df['label'].isin(selected_classes)]
    other_df=df[df['label'].isin(other_classes)]
    return selected_df,other_df
