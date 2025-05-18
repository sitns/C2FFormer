import os

import h5py
import numpy as np

def get_datasets_path_MAR(data_dir,name,rate):
    train_set_path = os.path.join(data_dir, "train.h5")
    val_set_path = os.path.join(data_dir, "val.h5")
    test_set_path = os.path.join(data_dir, "test.h5")

    with h5py.File(train_set_path, "r") as hf:
        train_X_ori = hf["X_ori"][:]
    with h5py.File(val_set_path, "r") as hf:
        val_X_ori_arr = hf["X_ori"][:]
    prepared_train_set = train_X_ori
    prepared_val_ori_arr = val_X_ori_arr

    with h5py.File(test_set_path, "r") as hf:
        test_X_ori_arr = hf["X_ori"][:]  # need test_X_ori_arr to calculate MAE and MSE
    
    # test_indicating_arr = ~np.isnan(test_X_ori_arr) ^ ~np.isnan(test_X_arr)
    test_X_ori_arr = np.nan_to_num(test_X_ori_arr)
    if name =='beijing' or name =='pedestrian':
        prepared_train_set = fill_nan_with_nearest_vectorized(prepared_train_set)
        prepared_val_ori_arr = fill_nan_with_nearest_vectorized(prepared_val_ori_arr)
    if name != 'pedestrian':
        if rate =='_rate0.9':
            test_X_arr = create_fixed_interval_missing(test_X_ori_arr, missing_rate=0.9)
            val_X_arr = create_fixed_interval_missing(val_X_ori_arr, missing_rate=0.9)
        if rate =='_rate0.5':
            test_X_arr = create_fixed_interval_missing(test_X_ori_arr, missing_rate=0.5)
            val_X_arr = create_fixed_interval_missing(val_X_ori_arr, missing_rate=0.5)
        if rate =='_rate0.1':
            test_X_arr = create_fixed_interval_missing(test_X_ori_arr, missing_rate=0.1)
            val_X_arr = create_fixed_interval_missing(val_X_ori_arr, missing_rate=0.1)
        prepared_val_arr = val_X_arr
        test_indicating_arr = ~np.isnan(test_X_ori_arr) ^ ~np.isnan(test_X_arr)
    if name =='pedestrian':
        test_X_ori_arr = test_X_ori_arr.transpose(0, 2, 1).astype(np.float32)
        val_X_ori_arr = val_X_ori_arr.transpose(0, 2, 1).astype(np.float32)
        if rate =='_rate0.9':
            test_X_arr = create_fixed_interval_missing(test_X_ori_arr, missing_rate=0.9)
            val_X_arr = create_fixed_interval_missing(val_X_ori_arr, missing_rate=0.9)
        if rate =='_rate0.5':
            test_X_arr = create_fixed_interval_missing(test_X_ori_arr, missing_rate=0.5)
            val_X_arr = create_fixed_interval_missing(val_X_ori_arr, missing_rate=0.5)
        if rate =='_rate0.1':
            test_X_arr = create_fixed_interval_missing(test_X_ori_arr, missing_rate=0.1)
            val_X_arr = create_fixed_interval_missing(val_X_ori_arr, missing_rate=0.1)
        prepared_val_arr = val_X_arr
        test_indicating_arr = ~np.isnan(test_X_ori_arr) ^ ~np.isnan(test_X_arr)
        prepared_train_set = prepared_train_set.transpose(0, 2, 1).astype(np.float32)
        prepared_val_ori_arr = prepared_val_ori_arr.transpose(0, 2, 1).astype(np.float32)

    return (
        prepared_train_set,
        prepared_val_arr,
        prepared_val_ori_arr,
        test_X_arr,
        test_X_ori_arr,
        test_indicating_arr,
    )

def create_fixed_interval_missing(tensor, missing_rate=0.9):
    """
    按固定间隔从tensor中抽取数据，其余位置填充NaN
    
    Args:
        tensor: 输入张量，形状为(batch_size, sequence_length, n_features)
        missing_rate: 缺失率，范围[0,1]
        
    Returns:
        result_tensor: 处理后的张量，与输入形状相同，未抽取位置填充NaN
    """
    import numpy as np
    
    # 复制输入张量
    result = tensor.copy()
    
    # 对于特别小的missing_rate，我们需要特殊处理
    if missing_rate < 0.5:
        # 对于低缺失率，我们反转思路：先标记要缺失的点
        # 对于0.1的缺失率，大约每10个点中缺失1个
        k_missing = max(int(1 / missing_rate), 2)
        
        # 创建mask，初始全为True（保留所有点）
        mask = np.ones_like(tensor, dtype=bool)
        
        # 每隔k_missing个点设置一个False（缺失）
        mask[:, ::k_missing, :] = False
    else:
        # 常规情况：标记要保留的点
        # 计算保留点之间的间隔(每隔k个点保留1个点)
        k = max(int(1 / (1 - missing_rate)), 2)
        
        # 创建mask，初始全为False（缺失所有点）
        mask = np.zeros_like(tensor, dtype=bool)
        
        # 每隔k个点设置一个True（保留）
        mask[:, ::k, :] = True
    
    # 将未被选中的位置设为NaN
    result[~mask] = np.nan
    
    return result
def get_datasets_path(data_dir,name):
    train_set_path = os.path.join(data_dir, "train.h5")
    val_set_path = os.path.join(data_dir, "val.h5")
    test_set_path = os.path.join(data_dir, "test.h5")

    
    with h5py.File(train_set_path, "r") as hf:
        # train_X_arr = hf["X"][:]
        train_X_ori = hf["X_ori"][:]
    with h5py.File(val_set_path, "r") as hf:
        val_X_arr = hf["X"][:]
        val_X_ori_arr = hf["X_ori"][:]
    prepared_train_set = train_X_ori
    prepared_val_arr = val_X_arr 
    prepared_val_ori_arr = val_X_ori_arr

    with h5py.File(test_set_path, "r") as hf:
        test_X_arr = hf["X"][:]
        test_X_ori_arr = hf["X_ori"][:]  # need test_X_ori_arr to calculate MAE and MSE

    test_indicating_arr = ~np.isnan(test_X_ori_arr) ^ ~np.isnan(test_X_arr)
    test_X_ori_arr = np.nan_to_num(test_X_ori_arr)
    if name =='beijing' or name =='pedestrian':
        prepared_train_set = fill_nan_with_nearest_vectorized(prepared_train_set)
        prepared_val_ori_arr = fill_nan_with_nearest_vectorized(prepared_val_ori_arr)
    if name =='pedestrian':
        prepared_train_set = prepared_train_set.transpose(0, 2, 1).astype(np.float32)
        prepared_val_arr = prepared_val_arr.transpose(0, 2, 1).astype(np.float32)
        prepared_val_ori_arr = prepared_val_ori_arr.transpose(0, 2, 1).astype(np.float32)
        test_X_arr = test_X_arr.transpose(0, 2, 1).astype(np.float32)
        test_X_ori_arr = test_X_ori_arr.transpose(0, 2, 1).astype(np.float32)
        test_indicating_arr = test_indicating_arr.transpose(0, 2, 1)
    return (
        prepared_train_set,
        prepared_val_arr,
        prepared_val_ori_arr,
        test_X_arr,
        test_X_ori_arr,
        test_indicating_arr,
    )

def fill_nan_with_nearest_vectorized(arr):
    """
    使用向量化操作实现的NaN值填充函数
    
    Args:
        arr: numpy数组，形状为(batch_size, sequence_length, n_features)
        
    Returns:
        filled_arr: 填充后的数组，与输入形状相同
    """
    filled_arr = arr.copy()
    
    for b in range(arr.shape[0]):
        for f in range(arr.shape[2]):
            seq = arr[b, :, f]
            mask = np.isnan(seq)
            
            if np.any(mask):
                # 获取非NaN值的索引
                valid_indices = np.where(~mask)[0]
                
                if len(valid_indices) > 0:
                    # 创建索引网格
                    idx_grid = np.arange(len(seq))
                    # 计算到所有非NaN值的距离矩阵
                    distances = np.abs(idx_grid[:, np.newaxis] - valid_indices)
                    # 找到最近的非NaN值
                    nearest_idx = valid_indices[np.argmin(distances, axis=1)]
                    # 填充NaN值
                    filled_arr[b, mask, f] = arr[b, nearest_idx[mask], f]
                else:
                    filled_arr[b, :, f] = 0
                    
    return filled_arr

