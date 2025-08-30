from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from model import get_model
import argparse
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from datasets import ECGSequence

def get_multilabel_weights(y_train):
    """为每个类别计算独立的类别权重"""
    class_weights = {}
    for class_idx in range(y_train.shape[1]):
        classes = np.unique(y_train[:, class_idx])
        # 处理全负样本的情况
        if len(classes) == 1 and classes[0] == 0:
            print(f"警告: 类别 {class_idx} ({ORIGINAL_LABELS[class_idx]}) 无正样本，跳过权重计算")
            class_weights[class_idx] = {0: 1.0, 1: 0.0}  # 手动设置默认权重
            continue
        
        # 计算权重
        weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train[:, class_idx]
        )
        # 构建权重字典（动态适应存在的类别）
        weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
        
        # 确保包含0和1键
        if 0 not in weight_dict:
            weight_dict[0] = 1.0  # 默认负类权重
        if 1 not in weight_dict:
            weight_dict[1] = 1.0  # 默认正类权重
            
        class_weights[class_idx] = weight_dict
    return class_weights

# 定义类别名称（需添加到代码中）
ORIGINAL_LABELS = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']

if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('path_to_hdf5', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_csv', type=str,
                        help='path to csv file containing annotations')
    parser.add_argument('--val_split', type=float, default=0.02,
                        help='number between 0 and 1 determining how much of'
                             ' the data is to be used for validation. The remaining '
                             'is used for validation. Default: 0.02')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    args = parser.parse_args()

    full_df = pd.read_csv(args.path_to_csv)
    y_full = full_df[['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']].values
    
    # 计算类别权重
    class_weights_dict = get_multilabel_weights(y_full)
    
    # 转换为Keras需要的格式：{class_index: weight}
    # 注意：这里假设类别顺序与ORIGINAL_LABELS一致
    class_weights = {
        i: class_weights_dict[i][1]  # 取正类权重
        for i in range(len(class_weights_dict))
    }

    # Optimization settings
    loss = 'binary_crossentropy'
    lr = 0.001
    batch_size = 32
    opt = Adam(lr)
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]

    train_seq, valid_seq = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split)

    # If you are continuing an interrupted section, uncomment line bellow:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    model = get_model(train_seq.n_classes)
    model.compile(loss=loss, optimizer=opt)
    # Create log
    callbacks += [TensorBoard(log_dir='./logs', write_graph=False),
                  CSVLogger('training.log', append=False)]  # Change append to true if continuing training
    # Save the BEST and LAST model
    callbacks += [
    ModelCheckpoint('./backup_model_last.h5'),  # 移除save_format参数
    ModelCheckpoint('./backup_model_best.h5', save_best_only=True)]

    # Train neural network
    history = model.fit(train_seq,
                        epochs=10,
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        validation_data=valid_seq,
                        verbose=1,
                        class_weight=class_weights)
    # Save final result
    model.save("./final_model.hdf5")


