# 提取ptbxl_database.csv文件中的scp_codes字段中的所有字段

import pandas as pd
import ast

# 加载数据
file_path = './ptbxl_database.csv'
df = pd.read_csv(file_path)

# 提取所有scp_codes
scp_codes_column = df['scp_codes']

# 创建一个空集合，用来存储所有的字段
all_fields = set()

# 遍历每一行，提取scp_codes中的字段
for scp_code_str in scp_codes_column:
    # 如果scp_codes字段有值
    if pd.notna(scp_code_str):
        try:
            # 将字符串转换为字典
            scp_code_dict = ast.literal_eval(scp_code_str)
            # 将字典的键加入集合（集合去重）
            all_fields.update(scp_code_dict.keys())
        except Exception as e:
            print(f"Error processing scp_codes: {scp_code_str} - {str(e)}")

# 显示所有字段
all_fields = sorted(all_fields)
print(all_fields)

# 统计每个病症对应字段的数量，并且在字典中后面的数值 > 50.0

# 首先解析 scp_codes 列的数据，将其转化为字典形式进行处理
def parse_scp_codes(scp_codes_str):
    try:
        return ast.literal_eval(scp_codes_str)
    except:
        return {}

# 提取每个病症字段并检查值大于50.0的情况
def count_fields_for_diseases(df, fields_to_check):
    counts = {field: 0 for field in fields_to_check}  # 初始化计数器

    # 遍历数据集中的每一行，检查 scp_codes 字段
    for scp_codes_str in df['scp_codes'].dropna():
        scp_codes_dict = parse_scp_codes(scp_codes_str)

        # 对每个病症字段进行检查
        for field, threshold in fields_to_check.items():
            if field == 'RBBB' and ('RBBB' in scp_codes_dict or 'CRBBB' in scp_codes_dict) and any(scp_codes_dict.get(code, 0) > threshold for code in ['RBBB', 'CRBBB']):
                counts['RBBB'] += 1
            elif field == 'LBBB' and ('LBBB' in scp_codes_dict or 'CLBBB' in scp_codes_dict or 'ILBBB' in scp_codes_dict) and any(scp_codes_dict.get(code, 0) > threshold for code in ['LBBB', 'CLBBB', 'ILBBB']):
                counts['LBBB'] += 1
            elif field in scp_codes_dict and scp_codes_dict[field] >= threshold:
                counts[field] += 1
    return counts

# 定义需要统计的字段和对应的阈值
fields_to_check = {
    '1AVB': 50.0,
    'RBBB': 50.0,
    'LBBB': 50.0,
    'SBRAD': 50.0,
    'AFIB': 50.0,
    'STACH': 50.0,  # 这里我们默认ST段异常类需要对其多个字段进行检查（STACH, STE_, STD_）
    'NORM': 50.0,
    'SR': 0.0
}

# 统计结果
counts = count_fields_for_diseases(df, fields_to_check)
print(counts)
