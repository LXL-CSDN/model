import h5py, numpy as np

CANDIDATE_KEYS = [
    "label", "labels", "diagnosis", "diagnoses",
    "diagnosis_code", "diagnosis_codes",
    "rhythm", "arrhythmia", "y", "ann", "annotation"
]

SEP_CHARS = [",", ";", " "]  # 多标签常见分隔符

def _to_py(obj):
    """把 h5py 返回的各种类型统一成 Python 基本类型（str/list/int 等）"""
    if obj is None:
        return None
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    if isinstance(obj, (np.bytes_, np.ndarray)) and obj.dtype.kind in ("S", "U"):
        # 字符串数组
        try:
            return [x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in obj.tolist()]
        except Exception:
            return str(obj)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj.item()
        return obj.tolist()
    return obj

def _split_labels(val):
    """统一把原始值转成 list[str]。"""
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        out = []
        for v in val:
            if v is None:
                continue
            if isinstance(v, (int, float)):
                out.append(str(v))
            else:
                s = str(v).strip()
                if not s:
                    continue
                # 再按分隔符拆一次
                for sep in SEP_CHARS:
                    if sep in s and not s.isnumeric():
                        out.extend([x.strip() for x in s.split(sep) if x.strip()])
                        break
                else:
                    out.append(s)
        return list(dict.fromkeys(out))  # 去重且保持顺序
    # 单值
    s = str(val).strip()
    if not s:
        return []
    for sep in SEP_CHARS:
        if sep in s and not s.isnumeric():
            return [x.strip() for x in s.split(sep) if x.strip()]
    return [s]

def extract_labels_from_h5(path):
    """
    返回一个 dict:
      {
        "record_id": 文件名不含后缀,
        "label_raw": 原始标签字符串/列表（转成; 连接便于看）,
        "label_list": 规范化后的字符串列表,
    可选:
        "codes": 若能判定为纯数字/编码则也放一份（字符串列表）
      }
    """
    import os
    rec_id = os.path.splitext(os.path.basename(path))[0]
    label_list = []
    label_raw = None
    codes = []

    with h5py.File(path, "r") as f:
        # 1) 文件属性
        for k in CANDIDATE_KEYS:
            if k in f.attrs:
                val = _to_py(f.attrs[k])
                label_raw = val if label_raw is None else label_raw
                label_list = _split_labels(val)
                break

        # 2) 顶层同名 dataset
        if not label_list:
            for k in CANDIDATE_KEYS:
                if k in f.keys() and isinstance(f[k], h5py.Dataset):
                    val = _to_py(f[k][()])
                    label_raw = val if label_raw is None else label_raw
                    label_list = _split_labels(val)
                    break

        # 3) diagnosis 分组
        if not label_list:
            if "diagnosis" in f.keys() and isinstance(f["diagnosis"], h5py.Group):
                g = f["diagnosis"]
                # 常见：code(s) / text(s)
                for k in ["codes", "code", "labels", "label", "text", "texts"]:
                    if k in g.keys():
                        val = _to_py(g[k][()])
                        label_raw = val if label_raw is None else label_raw
                        label_list = _split_labels(val)
                        break
                # 或者写在 group 的 attrs
                if not label_list:
                    for k in CANDIDATE_KEYS + ["code", "codes", "text", "texts"]:
                        if k in g.attrs:
                            val = _to_py(g.attrs[k])
                            label_raw = val if label_raw is None else label_raw
                            label_list = _split_labels(val)
                            break

        # 4) 粗略判断哪些像“编码”
        for v in label_list:
            s = v.replace("+", "").replace("-", "")
            if s.isnumeric():
                codes.append(v)

    return {
        "record_id": rec_id,
        "label_raw": ";".join(label_list) if isinstance(label_raw, (list, tuple)) else (label_raw if label_raw is not None else ""),
        "label_list": label_list,
        "codes": codes
    }
