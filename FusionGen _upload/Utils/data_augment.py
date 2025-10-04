
import numpy as np
from scipy.signal import hilbert
import scipy
# Add monkey patch for scipy.angle which might be missing in newer scipy versions
scipy.angle = np.angle
from pyhht import EMD
from scipy.interpolate import interp1d
import pywt

def data_aug(data, labels, subjects, target=None, flag_aug=None, dataset='BNCI2014002'):
    """
    Perform data augmentation based on the specified augmentation technique.
    
    Args:
        data: Input EEG data with shape (samples * n_channels * size)
        labels: Corresponding labels for the data
        subjects: Subject IDs for the data
        target: Target subject ID for subject-specific augmentation (default: None)
        flag_aug: Augmentation technique to use (default: None)
        dataset: Dataset name for dataset-specific configurations (default: 'BNCI2014002')
        
    Returns:
        augmented_data: Augmented EEG data
        augmented_labels: Labels corresponding to the augmented data
        augmented_subjects: Subject IDs corresponding to the augmented data
    """
    size = data.shape[2]
    n_channels = data.shape[1]
    data_out = data
    labels_out = labels
    subjects_out = subjects

    if flag_aug == 'Scale':
        mult_data_add, labels_out = data_mult_f(data, labels, size, n_channels=n_channels)
        data_out = mult_data_add
        subjects_out = np.concatenate((subjects_out, subjects), axis=0)
    elif flag_aug == 'Noise':
        noise_data_add, labels_out = data_noise_f(data, labels, size, n_channels=n_channels)
        data_out = noise_data_add
    elif flag_aug == 'Flip':
        neg_data_add, labels_out = data_neg_f(data, labels, size, n_channels=n_channels)
        data_out = neg_data_add
    elif flag_aug == 'Fshift':
        data_out, labels_out = freq_mod_f(data, labels, size, n_channels=n_channels)
        subjects_out = np.concatenate((subjects_out, subjects), axis=0)
    elif flag_aug == 'Cutdown_Resize':
        data_out = cutdown_resize(data, n_segments=10)
    elif flag_aug == 'CR':
        data_out, labels_out = CR(data, labels, dataset=dataset)
    elif flag_aug == 'DWTaug':
        data_out, labels_out, subjects_out = DWTAug(data, labels, subjects, target)
    elif flag_aug == 'HHTaug':
        data_out, labels_out, subjects_out = HHTAug(data, labels, subjects, target)
    elif flag_aug == 'VAE':
        data_out, labels_out = cvae(data, labels)

    return data_out, labels_out, subjects_out


def data_noise_f(data, labels, size, n_channels=15):
    """
    Add Gaussian noise to the EEG data.
    
    Args:
        data: Input EEG data
        labels: Corresponding labels
        size: Time dimension size
        n_channels: Number of channels (default: 15)
        
    Returns:
        Augmented data with noise added
        Corresponding labels
    """
    new_data = []
    new_labels = []
    noise_mod_val = 2
    
    for i in range(len(labels)):
        if labels[i] >= 0:
            stddev_t = np.std(data[i])
            rand_t = np.random.rand(data[i].shape[0], data[i].shape[1]) - 0.5
            to_add_t = rand_t * stddev_t / noise_mod_val
            data_t = data[i] + to_add_t
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, n_channels, size])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


def data_mult_f(data, labels, size, n_channels=15):
    """
    Apply scaling transformation to the EEG data.
    
    Args:
        data: Input EEG data
        labels: Corresponding labels
        size: Time dimension size
        n_channels: Number of channels (default: 15)
        
    Returns:
        Augmented data with scaling applied
        Corresponding labels
    """
    new_data = []
    new_labels = []
    mult_mod = 0.1
    
    # Apply scaling with both increase and decrease factors
    for factor in [1 + mult_mod, 1 - mult_mod]:
        for i in range(len(labels)):
            if labels[i] >= 0:
                data_t = data[i] * factor
                new_data.append(data_t)
                new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, n_channels, size])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


def data_neg_f(data, labels, size, n_channels=15):
    """
    Apply amplitude reflection to the EEG data.
    
    Args:
        data: Input EEG data
        labels: Corresponding labels
        size: Time dimension size
        n_channels: Number of channels (default: 15)
        
    Returns:
        Augmented data with amplitude reflection
        Corresponding labels
    """
    new_data = []
    new_labels = []
    
    for i in range(len(labels)):
        if labels[i] >= 0:
            # Apply amplitude reflection and shift to maintain non-negative values
            data_t = -1 * data[i]
            data_t = data_t - np.min(data_t)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, n_channels, size])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


def freq_mod_f(data, labels, size, n_channels=15):
    """
    Apply frequency modulation to the EEG data.
    
    Args:
        data: Input EEG data
        labels: Corresponding labels
        size: Time dimension size
        n_channels: Number of channels (default: 15)
        
    Returns:
        Augmented data with frequency shifts
        Corresponding labels
    """
    new_data = []
    new_labels = []
    freq_mod = 0.05
    
    # Apply both negative and positive frequency shifts
    for shift in [-freq_mod, freq_mod]:
        for i in range(len(labels)):
            if labels[i] >= 0:
                shifted_data = freq_shift(data[i], shift, num_channels=n_channels)
                new_data.append(shifted_data)
                new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, n_channels, size])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


def freq_shift(x, f_shift, num_channels, dt=1/250):
    """
    Apply frequency shift to the EEG data using Hilbert transform.
    
    Args:
        x: Input EEG data for a single sample
        f_shift: Frequency shift amount
        num_channels: Number of channels
        dt: Time step (default: 1/250)
        
    Returns:
        Frequency-shifted EEG data
    """
    len_x = x.shape[-1]
    padding_len = 2 ** nextpow2(len_x)
    padding = np.zeros((num_channels, padding_len - len_x))

    # Pad along the time axis
    with_padding = np.concatenate([x, padding], axis=-1)
    hilb_T = hilbert(with_padding, axis=-1)

    # Create frequency shift function
    t = np.arange(padding_len) * dt
    shift_func = np.exp(2j * np.pi * f_shift * t)

    # Apply frequency shift and crop back to original length
    shifted_sig = np.zeros_like(x)
    for i in range(num_channels):
        shifted = (hilb_T[i] * shift_func)[:len_x].real
        shifted_sig[i] = shifted

    return shifted_sig


def nextpow2(x):
    """
    Calculate the next power of 2 greater than or equal to the input.
    
    Args:
        x: Input value
        
    Returns:
        Integer representing the next power of 2
    """
    return int(np.ceil(np.log2(np.abs(x))))


def cutdown_resize(eeg_data, n_segments=10):
    """
    Process EEG data by randomly dropping a segment and resizing to original length.

    Args:
        eeg_data (np.ndarray): Input EEG data with shape (batchsize, channel, time)
        n_segments (int): Number of segments to split the time axis into

    Returns:
        np.ndarray: Processed EEG data with same shape as input
    """
    batchsize, channels, time_len = eeg_data.shape
    processed_data = np.zeros_like(eeg_data)

    # Validate parameters
    if n_segments < 2:
        raise ValueError("n_segments must be at least 2")
    if (time_len - 1) < (n_segments - 1):
        raise ValueError(f"Cannot split {time_len} time points into {n_segments} segments")

    for batch_idx in range(batchsize):
        # 1. Randomly split time axis into segments
        split_points = np.sort(np.random.choice(range(1, time_len), size=n_segments - 1, replace=False))
        split_indices = [0] + split_points.tolist() + [time_len]

        # 2. Create segment boundaries and select one to discard
        segments = [(split_indices[i], split_indices[i + 1]) for i in range(len(split_indices) - 1)]
        discard_idx = np.random.randint(n_segments)
        remaining_segments = [seg for i, seg in enumerate(segments) if i != discard_idx]

        # 3. Concatenate remaining segments and resize if needed
        concatenated = np.concatenate([eeg_data[batch_idx, :, start:end] for start, end in remaining_segments], axis=1)
        current_length = concatenated.shape[1]

        if current_length == time_len:
            resampled = concatenated
        else:
            # Use linear interpolation to resize to original length
            resampled = np.zeros((channels, time_len))
            x_old = np.arange(current_length)
            x_new = np.linspace(0, current_length - 1, time_len)

            for ch in range(channels):
                interpolator = interp1d(x_old, concatenated[ch], kind='linear', assume_sorted=True)
                resampled[ch] = interpolator(x_new)

        processed_data[batch_idx] = resampled

    return processed_data

def leftrightflipping_transform(X, labels, left_mat, right_mat):
    """
    Apply left-right flipping transformation to EEG channels.

    Args:
        X: Input EEG data with shape (num_samples, num_channels, num_timesamples)
        labels: Corresponding labels (not used in this function)
        left_mat: List of left brain channel indices
        right_mat: List of right brain channel indices (in corresponding order to left_mat)

    Returns:
        transformedX: Transformed EEG data with left-right flipped channels
    """
    num_samples, num_channels, num_timesamples = X.shape
    transformedX = np.zeros_like(X)
    
    # Create look-up dictionaries for faster index mapping
    left_to_right = {ch: right_mat[i] for i, ch in enumerate(left_mat)}
    right_to_left = {ch: left_mat[i] for i, ch in enumerate(right_mat)}
    
    for ch in range(num_channels):
        if ch in left_to_right:
            # Left channel maps to corresponding right channel
            transformedX[:, ch, :] = X[:, left_to_right[ch], :]
        elif ch in right_to_left:
            # Right channel maps to corresponding left channel
            transformedX[:, ch, :] = X[:, right_to_left[ch], :]
        else:
            # Midline channels remain unchanged
            transformedX[:, ch, :] = X[:, ch, :]

    return transformedX


def CR(data, labels, dataset='BNCI2014001'):
    """
    Apply cross-hemispheric data augmentation based on dataset-specific channel configurations.
    
    Args:
        data: Input EEG data
        labels: Corresponding labels
        dataset: Dataset name to determine channel configurations (default: 'BNCI2014001')
        
    Returns:
        Augmented data with cross-hemispheric transformation
        Corresponding labels (possibly flipped for binary classification tasks)
    """
    # Dataset-specific channel configurations
    config = {
        'BNCI2014001': {
            'left_mat': [1, 2, 6, 7, 8, 13, 14, 18],
            'right_mat': [5, 4, 12, 11, 10, 17, 16, 20],
            'flip_labels': True
        },
        'BNCI2014002': {
            'left_mat': [0, 3, 4, 5, 6, 12],
            'right_mat': [2, 11, 10, 9, 8, 14],
            'flip_labels': False
        },
        'Zhou2016': {
            'left_mat': [0, 2, 5, 8, 11],
            'right_mat': [1, 4, 7, 10, 13],
            'flip_labels': True
        }
    }
    
    # Get configuration for the specified dataset
    ds_config = config.get(dataset, config['BNCI2014001'])
    
    # Apply left-right flipping transformation
    aug_train_x = leftrightflipping_transform(data, labels, ds_config['left_mat'], ds_config['right_mat'])
    
    # Copy labels and flip if required by dataset
    aug_label = labels.copy()
    if ds_config['flip_labels']:
        label_0mask = labels == 0
        label_1mask = labels == 1
        aug_label[label_0mask] = 1
        aug_label[label_1mask] = 0
    
    return aug_train_x, aug_label

def get_src_and_tar(data,labels,subjects,target):
    subject_mask = subjects == target
    data_src = data[~ subject_mask]
    data_tar = data[subject_mask]
    labels_src = labels[~ subject_mask]
    labels_tar = labels[subject_mask]
    subjects_src = subjects[~ subject_mask]
    subjects_tar = subjects[subject_mask]
    return data_src, data_tar, labels_src, labels_tar, subjects_src, subjects_tar


def split_data_by_label(data, labels, subjects):
    # 获取所有唯一标签
    unique_labels = np.unique(labels)

    # 初始化两个分组的索引列表
    split1_indices = []
    split2_indices = []

    for label in unique_labels:
        # 获取当前标签的所有索引
        indices = np.where(labels == label)[0]

        # 计算分割点
        split_point = len(indices) // 2

        # 将索引分配到两个分组
        split1_indices.extend(indices[:split_point])
        split2_indices.extend(indices[split_point:])


    # 返回分割后的数据和标签
    return (
        data[split1_indices], labels[split1_indices],subjects[split1_indices],
        data[split2_indices], labels[split2_indices], subjects[split2_indices]
    )

def DWTAug(data, labels, subjects, target=None):
    """
    Apply Discrete Wavelet Transform based augmentation using cross-subject component mixing.
    
    Args:
        data: Input EEG data
        labels: Corresponding labels
        subjects: Subject identifiers
        target: Optional target subject index for source-target split
        
    Returns:
        Augmented data combining original and transformed samples
        Corresponding labels for augmented data
        Corresponding subject identifiers for augmented data
    """
    # Split data into source and target based on subject or label
    if target is None:
        data_src, labels_src, subjects_src, data_tar, labels_tar, subjects_tar = split_data_by_label(data, labels, subjects)
    else:
        data_src, data_tar, labels_src, labels_tar, subjects_src, subjects_tar = get_src_and_tar(data, labels, subjects, target)

    data_aug = []
    labels_aug = []
    subjects_aug = []
    
    # Process each source subject individually
    for subject_sel in np.unique(subjects_src):
        data_sel = data_src[subjects_src == subject_sel]
        
        # Process each label class
        for label in np.unique(labels):
            label_mask_src = labels_src[subjects_src == subject_sel] == label
            label_mask_tar = labels_tar == label
            
            if np.sum(label_mask_src) == 0 or np.sum(label_mask_tar) == 0:
                continue  # Skip if no samples for this label in either source or target
            
            # Perform wavelet decomposition for source and target data
            wavename = 'db4'
            Cs = pywt.wavedec(data_sel[label_mask_src], wavename, level=4)
            Ct = pywt.wavedec(data_tar[label_mask_tar], wavename, level=4)
            
            # Cross-subject component mixing and reconstruction
            # Source approximation + Target details
            Xt_aug = pywt.waverec([Cs[0], Ct[1], Ct[2], Ct[3], Ct[4]], wavename, 'smooth')
            # Target approximation + Source details
            Xs_aug = pywt.waverec([Ct[0], Cs[1], Cs[2], Cs[3], Cs[4]], wavename, 'smooth')
            
            # Ensure reconstructed signals match original length
            data_len = data_sel.shape[-1]
            Xt_aug = Xt_aug[:, :, :data_len]
            Xs_aug = Xs_aug[:, :, :data_len]
            
            # Append augmented samples to results
            data_aug.append(np.concatenate((Xt_aug, Xs_aug), axis=0))
            labels_aug.append(np.concatenate((labels_tar[label_mask_tar], labels_tar[label_mask_tar]), axis=0))
            subjects_aug.append(np.concatenate((subjects_tar[label_mask_tar], subjects_tar[label_mask_tar]), axis=0))

    # Combine results from all subjects and labels
    Data_aug = np.concatenate(data_aug)
    Labels_aug = np.concatenate(labels_aug)
    Subjects_aug = np.concatenate(subjects_aug)

    return Data_aug, Labels_aug, Subjects_aug

def HHTAug(data, labels, subjects, target=None):
    """
    Apply Hilbert-Huang Transform based augmentation using cross-subject IMF component mixing.
    
    Args:
        data: Input EEG data
        labels: Corresponding labels
        subjects: Subject identifiers
        target: Optional target subject index for source-target split
        
    Returns:
        Augmented data combining original and transformed samples
        Corresponding labels for augmented data
        Corresponding subject identifiers for augmented data
    """
    # Split data into source and target based on subject or label
    if target is None:
        data_src, labels_src, subjects_src, data_tar, labels_tar, subjects_tar = split_data_by_label(data, labels, subjects)
    else:
        data_src, data_tar, labels_src, labels_tar, subjects_src, subjects_tar = get_src_and_tar(data, labels, subjects, target)
    
    data_aug = []
    labels_aug = []
    subjects_aug = []
    
    # Process each source subject
    for subject_sel in np.unique(subjects_src):
        data_sel = data_src[subjects_src == subject_sel]
        
        # Ensure source and target data have matching dimensions
        min_samples = min(len(data_sel), len(data_tar))
        data_sel = data_sel[:min_samples] if len(data_sel) > min_samples else data_sel
        
        Xs_aug = []  # Target dominant + Source residual
        Xt_aug = []  # Source dominant + Target residual
        
        # Process each sample pair
        for s in range(min_samples):
            chns = []  # Channel data for target dominant
            chnt = []  # Channel data for source dominant
            
            for chn in range(data_sel.shape[1]):
                # Decompose signals into IMFs using EMD
                imfs_src = HHTFilter(data_sel[s][chn])
                imfs_tar = HHTFilter(data_tar[s][chn])
                
                # Separate into dominant components and residual
                components_most_src = list(range(len(imfs_src) - 1))  # All except last IMF
                components_less_src = [len(imfs_src) - 1]  # Last IMF (residual)
                components_most_tar = list(range(len(imfs_tar) - 1))
                components_less_tar = [len(imfs_tar) - 1]
                
                # Cross-subject component mixing and reconstruction
                # Source dominant components + Target residual
                chns_aug = np.sum(imfs_src[components_most_src] + imfs_tar[components_less_tar], axis=0)
                # Target dominant components + Source residual
                chnt_aug = np.sum(imfs_tar[components_most_tar] + imfs_src[components_less_src], axis=0)
                
                chns.append(chns_aug)
                chnt.append(chnt_aug)
                
            # Convert to arrays and append to results
            chns = np.array(chns)
            chnt = np.array(chnt)
            Xs_aug.append(chns)
            Xt_aug.append(chnt)
        
        # Convert lists to arrays
        Xt_aug = np.array(Xt_aug)
        Xs_aug = np.array(Xs_aug)
        
        # Ensure augmented data matches original length
        data_len = data_sel.shape[-1]
        Xt_aug = Xt_aug[:, :, :data_len]
        Xs_aug = Xs_aug[:, :, :data_len]
        
        # Append augmented samples
        data_aug.append(np.concatenate((Xt_aug, Xs_aug), axis=0))
        labels_aug.append(np.concatenate((labels_tar[:min_samples], labels_tar[:min_samples]), axis=0))
        subjects_aug.append(np.concatenate((subjects_tar[:min_samples], subjects_tar[:min_samples]), axis=0))
    
    # Combine all augmented data
    Data_aug = np.concatenate(data_aug)
    Labels_aug = np.concatenate(labels_aug)
    Subjects_aug = np.concatenate(subjects_aug)
    
    return Data_aug, Labels_aug, Subjects_aug

def HHTAnalysis(eegRaw, fs):
    # 进行EMD分解
    decomposer = EMD(eegRaw)
    # 获取EMD分解后的IMF成分
    imfs = decomposer.decompose()
    # 分解后的组分数
    n_components = imfs.shape[0]
    # 定义绘图，包括原始数据以及各组分数据
    fig, axes = plt.subplots(n_components + 1, 2, figsize=(10, 7), sharex=True, sharey=False)
    # 绘制原始数据
    axes[0][0].plot(eegRaw)
    # 原始数据的Hilbert变换
    eegRawHT = hilbert(eegRaw)
    # 绘制原始数据Hilbert变换的结果
    axes[0][0].plot(abs(eegRawHT))
    # 设置绘图标题
    axes[0][0].set_title('Raw Data')
    # 计算Hilbert变换后的瞬时频率
    instf, timestamps = tftb.processing.inst_freq(eegRawHT)
    # 绘制瞬时频率，这里乘以fs是正则化频率到真实频率的转换
    axes[0][1].plot(timestamps, instf * fs)
    # 计算瞬时频率的均值和中位数
    axes[0][1].set_title('Freq_Mean{:.2f}----Freq_Median{:.2f}'.format(np.mean(instf * fs), np.median(instf * fs)))

    # 计算并绘制各个组分
    for iter in range(n_components):
        # 绘制分解后的IMF组分
        axes[iter + 1][0].plot(imfs[iter])
        # 计算各组分的Hilbert变换
        imfsHT = hilbert(imfs[iter])
        # 绘制各组分的Hilber变换
        axes[iter + 1][0].plot(abs(imfsHT))
        # 设置图名
        axes[iter + 1][0].set_title('IMF{}'.format(iter))
        # 计算各组分Hilbert变换后的瞬时频率
        instf, timestamps = tftb.processing.inst_freq(imfsHT)
        # 绘制瞬时频率，这里乘以fs是正则化频率到真实频率的转换
        axes[iter + 1][1].plot(timestamps, instf * fs)
        # 计算瞬时频率的均值和中位数
        axes[iter + 1][1].set_title(
            'Freq_Mean{:.2f}----Freq_Median{:.2f}'.format(np.mean(instf * fs), np.median(instf * fs)))
    plt.tight_layout()
    plt.show()


def HHTFilter(eegRaw):
    """
    Apply Empirical Mode Decomposition (EMD) to decompose EEG signals into Intrinsic Mode Functions (IMFs).
    
    Args:
        eegRaw: Single-channel EEG signal
        
    Returns:
        imfs: Array of Intrinsic Mode Functions obtained from EMD decomposition
    """
    # Perform EMD decomposition
    decomposer = EMD(eegRaw)
    # Get IMF components from decomposition
    imfs = decomposer.decompose()
    return imfs
