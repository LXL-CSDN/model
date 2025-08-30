import wfdb
import numpy as np
import pandas as pd
import resampy
import h5py
import os
import ast
from scipy import signal
from tqdm import tqdm

# 配置参数
TARGET_SAMPLING_RATE = 400   # The sampling rate required by the model
TARGET_LENGTH = 4096         # Model input length
LEAD_ORDER = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
ORIGINAL_LABELS = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
BASE_DIR = '.' 

# 信号预处理函数（保持不变）
def preprocess_ecg(signal_12lead, fs):
    processed = []
    for lead in range(12):
        sig = signal_12lead[:, lead]
        # 1. 50Hz陷波滤波
        b, a = signal.iirnotch(50, 30, fs)
        sig = signal.filtfilt(b, a, sig)
        # 2. 五点平滑滤波
        sig = np.convolve(sig, np.ones(5)/5, mode='same')
        processed.append(sig)
    return np.array(processed).T

# 文件路径解析函数（增加容错）
def get_record_path(ecg_id):
    folder_num = ecg_id // 1000
    subfolder = f"{folder_num:02d}000"  # 修正为5位文件夹名（如'021000'）
    ecg_str = f"{ecg_id:05d}_lr"
    
    # 检查500Hz和100Hz数据路径
    paths_to_check = [
        os.path.join(BASE_DIR, 'records500', subfolder, ecg_str),
        os.path.join(BASE_DIR, 'records100', subfolder, ecg_str),
        os.path.join(BASE_DIR, 'records500', subfolder, f"{ecg_id:05d}"),  # 兼容无_lr后缀
        os.path.join(BASE_DIR, 'records100', subfolder, f"{ecg_id:05d}")
    ]
    
    for path in paths_to_check:
        if os.path.exists(path + '.hea'):
            return path
    raise FileNotFoundError(f"ECG ID {ecg_id} 文件不存在于: {paths_to_check}")

# 修改后的标签映射字典（数值型映射）
SCP_THRESHOLD_MAP = {
    '1dAVb': {'1AVB': 50.0},            # 1dAVb对应1AVB且数值>50
    'RBBB': {'RBBB': 50.0, 'CRBBB': 50.0}, 
    'LBBB': {'LBBB': 50.0, 'CLBBB': 50.0, 'ILBBB': 50.0},
    'SB': {'SBRAD': 50.0},
    'AF': {'AFIB': 50.0},
    'ST': {'STACH': 50.0, 'STE_': 50.0, 'STD_': 50.0}
}

def create_label(scp_codes_str):
    """根据SCP编码字典生成标签，仅当对应编码数值>阈值时标记"""
    label = np.zeros(len(ORIGINAL_LABELS), dtype=int)
    
    try:
        # 将字符串转换为字典（处理空值和非标准格式）
        scp_dict = ast.literal_eval(scp_codes_str.replace("'", "\""))  # 统一引号格式
    except:
        return label  # 解析失败返回全零标签
    
    for label_idx, label_name in enumerate(ORIGINAL_LABELS):
        # 获取当前类别对应的SCP编码及阈值
        code_rules = SCP_THRESHOLD_MAP.get(label_name, {})
        
        # 检查所有相关编码是否满足阈值
        for code, threshold in code_rules.items():
            code_value = scp_dict.get(code, 0.0)
            if code_value > threshold:
                label[label_idx] = 1
                break  # 只要有一个编码达标即标记
    return label

# 主处理流程（增加导联兼容性处理）
def main():
    df = pd.read_csv(os.path.join(BASE_DIR, 'ptbxl_database.csv'), index_col='ecg_id')
    all_records = df.index.tolist()

    signals = []
    labels = []
    error_log = []

    for ecg_id in tqdm(all_records):
        try:
            # 获取文件路径
            record_path = get_record_path(ecg_id)
            
            # 读取信号（强制指定导联名称）
            record = wfdb.rdrecord(record_path, physical=True, channel_names=LEAD_ORDER)
            original_signal = record.p_signal
            
            # 预处理流程
            processed = preprocess_ecg(original_signal, record.fs)
            
            # 重采样（优化性能）
            resampled = resampy.resample(
                processed.T,  # 转置为(12, length)以适应resampy
                sr_orig=record.fs,
                sr_new=TARGET_SAMPLING_RATE,
                axis=1
            ).T  # 转置回(length, 12)
            
            # 标准化长度
            if resampled.shape[0] >= TARGET_LENGTH:
                truncated = resampled[:TARGET_LENGTH, :]
            else:
                truncated = np.pad(resampled, ((0, TARGET_LENGTH - resampled.shape[0]), (0, 0)), mode='constant')
            
            # 标签生成（添加调试）
            scp_codes_str = df.loc[ecg_id, 'scp_codes']  # 获取完整字典字符串
            label = create_label(scp_codes_str)          # 传入完整字符串
        
            
            signals.append(truncated)
            labels.append(label)
            
        except Exception as e:
            error_log.append(f"ECG ID {ecg_id}: {str(e)}")
            continue

    # 保存结果
    with h5py.File('ptbxl_processed.hdf5', 'w') as hf:
        hf.create_dataset('tracings', data=np.array(signals))
    
    label_df = pd.DataFrame(labels, columns=ORIGINAL_LABELS)
    label_df.to_csv('ptbxl_labels.csv', index=False)
    
    # 输出日志
    with open('preprocess_errors.log', 'w') as f:
        f.write("\n".join(error_log))
    
    print(f"处理完成！有效数据量: {len(signals)}")
    print(f"错误日志已保存至: preprocess_errors.log")

if __name__ == '__main__':
    main()