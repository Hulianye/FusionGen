from mne.conftest import subjects_dir
from moabb.datasets import Zhou2016
from moabb.paradigms import LeftRightImagery
from moabb.paradigms import MotorImagery
import numpy as np
import mne
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.signal import resample
from scipy.linalg import fractional_matrix_power
class DATA:
    def __init__(self, data, label,subject):
        self.data = data
        self.label = label
        self.subject = subject


def resample_data(data, original_fs, target_fs):
    """
    将 (batchsize, channel, time) 的数据从 original_fs 降采样到 target_fs
    """
    batch, channel, time = data.shape
    new_length = int(round(time * target_fs / original_fs))

    # 将数据重塑为二维数组 (batch * channel, time)
    data_reshaped = data.reshape(-1, time)

    # 应用重采样
    resampled = resample(data_reshaped, new_length, axis=-1)

    # 恢复原始形状 (batch, channel, new_length)
    return resampled.reshape(batch, channel, new_length)


def download_data_Zhou2016():
    dataset = Zhou2016()

    # 使用 MotorImagery paradigm
    paradigm = MotorImagery()
    alldata = paradigm.get_data(dataset)
    data, labels_string, metadata = alldata
    data = data[:, :, :1000]
    subjects = metadata["subject"].values
    sessions = metadata["session"].values

    label_map = {
        "right_hand": 0,
        "left_hand": 1,
        "feet": 2,
    }

    # 将标签转换为数字
    labels = [label_map[label] for label in labels_string]
    labels = np.array(labels)

    save_dir = os.path.abspath('./data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, 'Zhou2016_data.npz')



    np.savez(
        file_path,
        data=data,
        labels=labels,
        subjects=subjects,
        sessions=sessions
    )
    return data, labels, subjects, sessions

def import_data_Zhou2016():
    loaded = np.load('./data/Zhou2016_data.npz',allow_pickle=True)
    data = loaded['data']
    labels = loaded['labels']
    subjects = loaded['subjects']
    sessions = loaded['sessions']
    # label_sel = [0, 1]
    # label_mask = np.isin(labels, label_sel)
    # data = data[label_mask]
    # labels = labels[label_mask]
    # subjects = subjects[label_mask]
    # sessions = sessions[label_mask]
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

def getdata_Zhou2016(data, labels, subjects, sessions,session_selected = None, subject_selected = None,val_subect_selected = None,train_num = 10):

    data, labels, subjects, sessions = select_data(data, labels, subjects, sessions, session_selected=session_selected,
                                                   subject_selected=subject_selected)

    # label_sel = [0,1]
    # label_mask = np.isin(labels, label_sel)
    # data = data[label_mask]
    # labels = labels[label_mask]
    # subjects = subjects[label_mask]
    # sessions = sessions[label_mask]

    df = pd.DataFrame({
        'data': list(data),
        'labels': labels,
        'subjects': subjects,
        'sessions': sessions
    })


    # 分离出训练集、验证集和测试集
    train_data = []
    train_labels = []
    train_subjects = []
    val_data = []
    val_labels = []
    val_subjects = []


    for (subject, label), group in df.groupby(['subjects', 'labels']):
        # 训练集前 5 个
        train_group = group.head(train_num)
        if len(group) > 0:
            train_data.append(np.array(train_group['data'].tolist()))  # (n, ch, time)
            train_labels.append(train_group['labels'].values)
            train_subjects.append(train_group['subjects'].values)

        # 验证集
        val_group = group.iloc[20:]
        if val_subect_selected is None:
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


def getdata_all_Zhou2016(session_selected = None, subject_selected = None):

    data, labels, subjects, sessions = download_data_Zhou2016()

    data,labels,subjects,sessions = select_data(data, labels, subjects, sessions,session_selected = session_selected, subject_selected = subject_selected)

    train_data,  test_data,train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    train_subjects = np.ones(train_labels.shape[0])
    test_subjects = np.ones(test_labels.shape[0])
    # # Train = DATA(limited_data, limited_labels,limited_subjects)
    Train = DATA(train_data, train_labels,train_subjects)
    Validation =DATA(test_data, test_labels,test_subjects)
    return Train, Validation

