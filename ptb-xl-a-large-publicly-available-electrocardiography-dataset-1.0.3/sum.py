import pandas as pd
label_df = pd.read_csv('ptbxl_labels.csv')
print("标签分布统计:")
print(label_df.sum())