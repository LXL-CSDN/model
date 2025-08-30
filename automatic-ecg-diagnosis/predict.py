import numpy as np
import pandas as pd
import warnings
import argparse
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from datasets import ECGSequence

# 新增参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('path_to_hdf5', type=str,
                      help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_model',
                      help='file containing training model.')
    parser.add_argument('--label_csv', type=str, required=True,
                      help='path to label csv file')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                      help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--output_file', default="./dnn_output.npy",
                      help='output csv file.')
    parser.add_argument('-bs', type=int, default=32,
                      help='Batch size.')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Classification threshold')
    return parser.parse_args()

# 新增指标计算函数
def calculate_metrics(y_true, y_pred, classes):
    results = {}
    for idx, class_name in enumerate(classes):
        results[class_name] = {
            "accuracy": accuracy_score(y_true[:, idx], y_pred[:, idx]),
            "recall": recall_score(y_true[:, idx], y_pred[:, idx]),
            "precision": precision_score(y_true[:, idx], y_pred[:, idx]),
            "f1": f1_score(y_true[:, idx], y_pred[:, idx])
        }
    return results

if __name__ == '__main__':
    args = parse_args()
    
    # 加载标签
    label_df = pd.read_csv(args.label_csv)
    target_classes = ['1dAVb', 'RBBB', 'LBBB']
    y_true = label_df[target_classes].values
    
    # 加载数据和模型
    seq = ECGSequence(args.path_to_hdf5, args.dataset_name, batch_size=args.bs)
    model = load_model(args.path_to_model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    
    # 预测并保存
    y_score = model.predict(seq, verbose=1)
    np.save(args.output_file, y_score)
    
    # 二值化预测结果
    y_pred = (y_score >= args.threshold).astype(int)
    
    # 计算并打印指标
    metrics = calculate_metrics(y_true, y_pred, target_classes)
    
    print("\n性能指标：")
    for class_name, scores in metrics.items():
        print(f"\n{class_name}:")
        print(f"  Accuracy: {scores['accuracy']:.4f}")
        print(f"  Recall:   {scores['recall']:.4f}")
        print(f"  Precision:{scores['precision']:.4f}")
        print(f"  F1-score: {scores['f1']:.4f}")
    
    print("\n分类报告：")
    print(classification_report(y_true, y_pred, target_names=target_classes))