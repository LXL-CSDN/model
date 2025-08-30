import wfdb
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import resample_poly
import resampy
import h5py
import os
import ast
from tqdm import tqdm

# ========= 配置 =========
TARGET_SAMPLING_RATE = 400
TARGET_LENGTH = 4096
# 使用 PTB-XL 的真实导联名（注意 aVR/aVL/aVF 小写 a）
LEAD_ORDER = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# 11 类（原 6 类 + 新 5 类）。如只想训练新增 5 类，可把前 6 个去掉。
TARGET_LABELS = [
    '2dAVB', '3dAVB', 'PAC', 'PVC', 'AFLT'        # 新增5类
]

BASE_DIR = '.'            # 指向包含 ptbxl_database.csv 的根目录
USE_FILENAME_FROM_CSV = True  # 更稳：用 df['filename_lr'] 构造路径

# ========= 标签映射（SCP -> 我们的标签） =========
# 每个标签对应若干 SCP 码，且只要有任一 SCP 码的分值 > 阈值即记为 1
SCP_THRESHOLD_MAP = {
    # 新增五类（覆盖常见同义/别名，按需再扩）
    '2dAVB': {'2AVB': 50.0, 'AVB2': 50.0, 'MOBITZ1': 50.0, 'MOBITZ2': 50.0, 'WENCK': 50.0},
    '3dAVB': {'CAVB': 50.0, '3AVB': 50.0, 'AVB3': 50.0, 'COMPAVB': 50.0},
    'PAC':   {'PAC': 50.0, 'SVPB': 50.0, 'SVPC': 50.0, 'SVEB': 50.0},
    'PVC':   {'PVC': 50.0, 'VPB': 50.0, 'VPC': 50.0},
    'AFLT':  {'AFLT': 50.0, 'AFL': 50.0},
}

# ========= 工具函数 =========
def parse_scp_dict(s):
    """ptbxl_database.csv 的 scp_codes 是字符串字典，这里解析成真字典"""
    if pd.isna(s):
        return {}
    try:
        # PTB-XL 原始就是单引号，ast.literal_eval 能直接解析，不必替换
        d = ast.literal_eval(s)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {}

def create_multilabel(scp_codes_str, label_names):
    scp = parse_scp_dict(scp_codes_str)
    y = np.zeros(len(label_names), dtype=np.int32)
    for i, lab in enumerate(label_names):
        rules = SCP_THRESHOLD_MAP.get(lab, {})
        for code, thr in rules.items():
            val = float(scp.get(code, 0.0))
            if val > thr:
                y[i] = 1
                break
    return y

def scale_to_1e4V(x, units):
    """
    把 WFDB 的物理量值统一到“以 1e-4 V 为单位”的标度。
    x: (T, 12) 的 float 数组（物理量）
    units: list[str]，每导联单位，如 'mV'/'V'/'uV'
    """
    # 允许混合单位；逐导联缩放
    u = units if isinstance(units, (list, tuple)) else [units] * x.shape[1]
    y = np.empty_like(x, dtype=np.float32)
    for ch in range(x.shape[1]):
        unit = (u[ch] or '').lower()
        if unit == 'v':
            s = 1e4
        elif unit in ('mv', 'mv.'):           # 偶见 'mV.' 之类
            s = 10.0
        elif unit in ('uv', 'µv', 'μv'):
            s = 0.01
        else:
            # 未知单位：PTB-XL 常见是 mV，这里按 mV 处理
            s = 10.0
        y[:, ch] = (x[:, ch] * s).astype(np.float32)
    return y

def bandstop_notch_50hz(x, fs):
    """对每导联做 50Hz 陷波 + 5 点滑动平均"""
    # 只在每条记录内设计一次滤波器
    b, a = signal.iirnotch(50, 30, fs=fs)
    y = signal.filtfilt(b, a, x, axis=0)
    # 5 点滑动平均
    kernel = np.ones(5, dtype=np.float32) / 5.0
    # 对每导联做一维卷积
    for ch in range(y.shape[1]):
        y[:, ch] = np.convolve(y[:, ch], kernel, mode='same')
    return y.astype(np.float32)

def resample_to_400hz(x, fs_orig):
    """把 (T, 12) 的信号重采样到 400Hz，优先 resample_poly"""
    if fs_orig == 400:
        return x
    if fs_orig == 500:
        # 400/500 = 4/5
        return resample_poly(x, up=4, down=5, axis=0).astype(np.float32)
    if fs_orig == 100:
        # 400/100 = 4
        return resample_poly(x, up=4, down=1, axis=0).astype(np.float32)
    # 其他采样率，fallback 到 resampy
    return resampy.resample(x.T, sr_orig=fs_orig, sr_new=TARGET_SAMPLING_RATE, axis=0).T.astype(np.float32)

def pad_or_trim(x, target_len=TARGET_LENGTH):
    """标准化长度到 (target_len, 12)"""
    T = x.shape[0]
    if T == target_len:
        return x
    if T > target_len:
        return x[:target_len, :]
    y = np.zeros((target_len, x.shape[1]), dtype=np.float32)
    y[:T, :] = x
    return y

def safe_rdrecord(path):
    """统一读取 WFDB 记录为物理量 (p_signal)、单位 sig_units"""
    rec = wfdb.rdrecord(path, physical=True, channel_names=LEAD_ORDER)
    # 有的 WFDB 版本字段名是 'sig_units'，有的是 'units'
    units = getattr(rec, 'sig_units', None) or getattr(rec, 'units', None) or ['mV'] * len(rec.sig_name)
    return rec.p_signal, rec.fs, units

# ========= 主流程 =========
def main():
    csv_path = os.path.join(BASE_DIR, 'ptbxl_database.csv')
    df = pd.read_csv(csv_path, index_col='ecg_id')

    # 采用 PTB-XL 自带的相对路径更稳：filename_lr（100Hz） 或 filename_hr（500Hz）
    path_series = None
    if USE_FILENAME_FROM_CSV:
        # 优先 500Hz；若缺失则用 100Hz
        if 'filename_hr' in df.columns and df['filename_hr'].notna().any():
            path_series = df['filename_hr'].fillna(df.get('filename_lr'))
        else:
            path_series = df['filename_lr']
        # 去掉开头的 './' 以便 os.path.join
        path_series = path_series.str.lstrip('./')
    else:
        # 兼容：仍用手写路径（不推荐）
        def get_record_path(ecg_id):
            folder_num = ecg_id // 1000
            subfolder = f"{folder_num:02d}000"  # '00000','01000',...
            ecg_str = f"{ecg_id:05d}_lr"
            candidates = [
                os.path.join(BASE_DIR, 'records500', subfolder, ecg_str),
                os.path.join(BASE_DIR, 'records100', subfolder, ecg_str),
                os.path.join(BASE_DIR, 'records500', subfolder, f"{ecg_id:05d}"),
                os.path.join(BASE_DIR, 'records100', subfolder, f"{ecg_id:05d}")
            ]
            for p in candidates:
                if os.path.exists(p + '.hea'):
                    return p
            raise FileNotFoundError(f"ECG ID {ecg_id} not found in {candidates}")
        path_series = df.index.to_series().apply(get_record_path)

    # 预创建 HDF5，边处理边写，避免爆内存
    n = len(df)
    h5_out = 'new_ptbxl_processed.hdf5'
    lab_out = 'new_ptbxl_labels.csv'
    err_log = 'new_preprocess_errors.log'

    # 提前拿标签
    y_all = np.zeros((n, len(TARGET_LABELS)), dtype=np.int8)
    ids = np.zeros((n,), dtype=np.int32)

    with h5py.File(h5_out, 'w') as hf, open(err_log, 'w') as elog:
        dset = hf.create_dataset(
            'tracings',
            shape=(n, TARGET_LENGTH, 12),
            dtype='float32',
            chunks=(1, TARGET_LENGTH, 12),
            compression='gzip',
            compression_opts=4
        )
        idset = hf.create_dataset('ecg_id', data=np.zeros((n,), dtype='int32'))

        for idx, (ecg_id, relpath) in enumerate(tqdm(path_series.items(), total=n)):
            try:
                # 读 WFDB
                fullpath = os.path.join(BASE_DIR, relpath)
                fullpath = fullpath[:-4] if fullpath.endswith('.dat') or fullpath.endswith('.hea') else fullpath
                sig, fs, units = safe_rdrecord(fullpath)   # (T, 12), fs, units(list)

                # 预处理：50Hz notch + 平滑
                sig = bandstop_notch_50hz(sig, fs)

                # 重采样到 400Hz
                sig = resample_to_400hz(sig, fs)

                # 幅度单位对齐到 1e-4V 标度
                sig = scale_to_1e4V(sig, units)

                # 长度标准化
                sig = pad_or_trim(sig, TARGET_LENGTH)

                # 保存一条
                dset[idx, :, :] = sig
                idset[idx] = int(ecg_id)
                ids[idx] = int(ecg_id)

                # 生成多标签
                scp_codes_str = df.loc[ecg_id, 'scp_codes']
                y_all[idx, :] = create_multilabel(scp_codes_str, TARGET_LABELS)

            except Exception as e:
                elog.write(f"ECG ID {ecg_id}: {repr(e)}\n")
                # 失败样本以 0 填充并保留默认 0 标签；也可选择跳过（但要记得同时删除对应行）
                continue

    # 输出标签 CSV（含 ecg_id 方便对齐）
    out_df = pd.DataFrame(y_all, columns=TARGET_LABELS)
    out_df.insert(0, 'ecg_id', ids)
    out_df['strat_fold'] = [int(df.loc[eid, 'strat_fold']) for eid in ids]
    if 'patient_id' in df.columns:
        out_df['patient_id'] = [int(df.loc[eid, 'patient_id']) for eid in ids]
    out_df.to_csv(lab_out, index=False)

    print(f"Done. Saved HDF5 to {h5_out}, labels to {lab_out}. Errors -> {err_log}")
    print(f"有效数据量(含失败样本的0填充行)：{len(out_df)}；"
          f"正例统计：\n{out_df[TARGET_LABELS].sum(axis=0)}")

if __name__ == '__main__':
    main()
