from mne.conftest import subjects_dir
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from moabb.paradigms import MotorImagery
import numpy as np
import mne
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
from scipy.linalg import fractional_matrix_power


class DATA:
    def __init__(self, data, label,subject):
        self.data = data
        self.label = label
        self.subject = subject

def download_data_BNCI2014_001():
    dataset = BNCI2014_001()

    # 使用 MotorImagery paradigm
    paradigm = MotorImagery()
    alldata = paradigm.get_data(dataset)
    data, labels_string, metadata = alldata
    data = data[:, :, :1000]
    subjects = metadata["subject"].values
    sessions = metadata["session"].values

    label_map = {
        "left_hand": 0,
        "right_hand": 1,
        "feet": 2,
        "tongue": 3
    }

    # 将标签转换为数字
    labels = [label_map[label] for label in labels_string]
    labels = np.array(labels)

    save_dir = os.path.abspath('./data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, 'BNCI2014_data.npz')

    np.savez(
        file_path,
        data=data,
        labels=labels,
        subjects=subjects,
        sessions=sessions
    )
    return data, labels, subjects,sessions

def import_data_BNCI2014_001():
    loaded = np.load('./data/BNCI2014_data.npz',allow_pickle=True)
    data = loaded['data']
    labels = loaded['labels']
    subjects = loaded['subjects']
    sessions = loaded['sessions']
    return data, labels, subjects, sessions

def select_data(data, labels, subjects, sessions,session_selected = None, subject_selected = None):

    if session_selected is not None:
        sessions_mask = (sessions == session_selected)
        data = data[sessions_mask]
        labels = labels[sessions_mask]
        subjects = subjects[sessions_mask]
        sessions = sessions[sessions_mask]

    if subject_selected is not None:
        subjects_mask = np.isin(subjects, subject_selected)
        data = data[subjects_mask]
        labels = labels[subjects_mask]
        subjects = subjects[subjects_mask]
        sessions = sessions[subjects_mask]
    return data, labels, subjects, sessions

def getdata(data, labels, subjects, sessions,session_selected = None, subject_selected = None,val_subect_selected = None,train_num = 10):

    data, labels, subjects, sessions = select_data(data, labels, subjects, sessions,session_selected = session_selected, subject_selected = subject_selected)

    df = pd.DataFrame({
        'data': list(data),
        'labels': labels,
        'subjects': subjects,
        'sessions': sessions
    })
    # # 进行分组
    # grouped = df.groupby(['subjects', 'labels']).apply(lambda x: x)

    # 分离出训练集、验证集和测试集
    train_data = []
    train_labels = []
    train_subjects = []
    val_data = []
    val_labels = []
    val_subjects = []

    for (subject, label), group in df.groupby(['subjects', 'labels']):
        # 训练集
        train_group = group.head(train_num)
        if len(group) > 0:
            train_data.append(np.array(train_group['data'].tolist()))  # (n, ch, time)
            train_labels.append(train_group['labels'].values)
            train_subjects.append(train_group['subjects'].values)

        # 验证集
        val_group = group.iloc[5:]
        if val_subect_selected is  None:
            val_data.append(np.array(val_group['data'].tolist()))
            val_labels.append(val_group['labels'].values)
            val_subjects.append(val_group['subjects'].values)
        elif subject == val_subect_selected:
            val_data.append(np.array(val_group['data'].tolist()))
            val_labels.append(val_group['labels'].values)
            val_subjects.append(val_group['subjects'].values)

    # 将数据转换为 numpy 数组
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_subjects = np.concatenate(train_subjects, axis=0)

    val_data = np.concatenate(val_data, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_subjects = np.concatenate(val_subjects, axis=0)

    Train = DATA(train_data, train_labels, train_subjects)
    Validation = DATA(val_data, val_labels, val_subjects)


    return Train, Validation

def EA(x,Ref = None,num=None): #x(bs,channel,point)

    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1])) #(bs,channel,channel)
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    numEA = x.shape[0]

    if Ref is not None:
       refEA = (Ref*num+refEA*numEA)/(num+numEA)

    sqrtRefEA = fractional_matrix_power(refEA, -0.5) + (0.00000001) * np.eye(x.shape[1])

    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])

    return XEA,refEA,numEA



def getdata_all(data, labels, subjects, sessions,session_selected = None, subject_selected = None):

    data,labels,subjects,sessions = select_data(data, labels, subjects, sessions,session_selected = session_selected, subject_selected = subject_selected)

    train_data,  test_data,train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    train_subjects = np.ones(train_labels.shape[0])
    test_subjects = np.ones(test_labels.shape[0])
    # # Train = DATA(limited_data, limited_labels,limited_subjects)
    Train = DATA(train_data, train_labels,train_subjects)
    Validation =DATA(test_data, test_labels,test_subjects)
    return Train, Validation


