import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import copy
import torch
import torch.nn as nn
from Config import parse_args
from datetime import datetime
from pathlib import Path
import time
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.metrics import Metric_Manager
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import json
import torch.nn.functional as F
from data.dataset import infarction_Dataset, ICH_Dataset
from model.OurNet import OurNet
from utils.loss import Loss, compute_ground_truth_adjacency
from sklearn.metrics import roc_auc_score
import random
from utils.tSNE import draw_tsne


def to0_1(tensor):
    bool_tensor = tensor > 0.5
    # 将布尔型张量转换为浮点型张量，True变为1.0，False变为0.0
    result_tensor = bool_tensor.float()
    return result_tensor

def split_dataset(dataset, test_size=0.2):
    """
    将数据集随机划分为训练集和测试集
    :param dataset: 原始数据集
    :param test_size: 测试集比例
    :return: 训练集和测试集的 Subset
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_size = int(np.floor(test_size * dataset_size))
    np.random.shuffle(indices)
    
    train_indices, test_indices = indices[test_size:], indices[:test_size]
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    return train_subset, test_subset
 
 
def k_fold_cross_validation_with_test(device, dataset, epochs, test_size=0.2, k_fold=5,
                                      batch_size=8, workers=2, print_freq=1, model_dir="./checkpoints/"):
    """
    将数据集划分为训练集和测试集，然后对训练集执行 k 折交叉验证
    :param device: 设备 (cuda/cpu)
    :param dataset: 原始数据集
    :param epochs: 每个 fold 的训练轮数
    :param test_size: 测试集比例
    :param k_fold: k 折交叉验证的折数
    :param batch_size: 批大小
    :param workers: 数据加载线程数
    :param print_freq: 日志打印频率
    """
    start = time.time()
    best_acc_overall = 0.
    epoch_acc_dict = {}
    
    train_metric = Metric_Manager(num_classes=1)
    valid_metric = Metric_Manager(num_classes=1)
    
    # 划分训练集和测试集
    train_dataset, test_dataset = split_dataset(dataset, test_size=test_size)
    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
    
    # 包装测试集 DataLoader
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # KFold 交叉验证
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=0)  # init KFold
    loss_fn = Loss(alpha=0.2, beta=1.0,).to(device)
    # loss_fn = nn.CrossEntropyLoss().to(device)  # 损失函数

    
    for fold, (train_index, valid_index) in enumerate(kf.split(train_dataset)):  # split
        k_train_fold = Subset(train_dataset, train_index)
        k_valid_fold = Subset(train_dataset, valid_index)
        print(f"Fold {fold}: Train size: {len(k_train_fold)}, Validation size: {len(k_valid_fold)}")
 
        # package type of DataLoader
        train_dataloader = torch.utils.data.DataLoader(dataset=k_train_fold, batch_size=batch_size, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(dataset=k_valid_fold, batch_size=batch_size, shuffle=False)
        model = OurNet(
            img_dim=(1, 20, 256, 256),  
            tab_dim=2,               # 假设表格数据有2个特征
            d_model=128,              # 统一特征维度
            nhead=4                   # 注意力头数
        ).to(device)
        # model = DLbase().to(device)
        
        # pos = 0
        # neg = 0
        # for batch_idx, (image, label, table) in enumerate(train_dataloader):
        #     pos += torch.sum(label == 1).item()
        #     neg += torch.sum(label == 0).item()
        # print(f"Fold {fold}: Positive samples: {pos}, Negative samples: {neg}")
        # # 计算正负样本比例
        # pos_ratio = neg / pos
        # weight = torch.FloatTensor([1, pos_ratio])
        # loss_fn = nn.CrossEntropyLoss().to(device)  # 损失函数
    
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-7, betas=(0.9, 0.98))
        total_params = sum(p.numel() for p in params)
        print('总参数个数:{}'.format(total_params))
 
        best_score = 0.
        best_epoch = 0
        best_acc = 0.
        best_recall = 0.
        best_precision = 0.
        best_specificity = 0.
        best_auc = 0.
        best_metrics = {}
        best_model_path = os.path.join(model_dir, f"main2_fold{fold}.pth")
        for e in range(1, epochs + 1):
            model.train()
            train_metric.reset()
            valid_metric.reset()
            
            total_train_loss = 0.0
            total_valid_loss = 0.0
            train_iterator = tqdm(train_dataloader, desc=f"Training Epoch {e}", unit="batch")
            all_train_labels = []
            all_train_preds = []
            for batch_idx, (image, label, table) in enumerate(train_iterator):
                # print(table, label)
                image = image.to(device)
                label = label.to(device)
                table = table.to(device)
                # print(image.shape, table.shape, label.shape, label)
                adj_pred, features, output, csdm_loss = model(image, table)
                adj_true = compute_ground_truth_adjacency(label)
                print(output, label)
                # draw_tsne(data=features, labels=label, path=f'./visual/tSNE_fold{fold}_batch{batch_idx}.png')

                optimizer.zero_grad()
                structure_loss, classification_loss_val = loss_fn(adj_pred, adj_true, features, label, output)
                print(csdm_loss, structure_loss, classification_loss_val)
                # total_loss = 0.1 * csdm_loss + 0.1 * structure_loss + classification_loss_val
                total_loss = classification_loss_val
                # loss = loss_fn(output, label)
                total_loss.backward()
                optimizer.step()
                total_train_loss += total_loss.item()
                
                train_metric.update(output, label)
                # Collect labels and predictions for AUC calculation
                all_train_labels.extend(label.cpu().numpy())
                all_train_preds.extend(F.softmax(output, dim=1)[:, 1].cpu().detach().numpy())  # Probabilities for class 1

            train_accuracy, train_recall, train_precision, train_specificity = train_metric.get_metrics()
            TP, TN, FP, FN = train_metric.get_matrix()
            total_train_loss /= len(train_index)
            
            # Calculate AUC for training (optional, can be removed if not needed)
            train_auc = roc_auc_score(all_train_labels, all_train_preds)
            
            if e % print_freq == 0:
                output_result = f"Epoch [{e}/{epochs}]: \n train_loss={total_train_loss:.3f}, \n accuracy={train_accuracy}, \n recall={train_recall}, \n precision={train_precision}, \n specificity={train_specificity}, \n AUC={train_auc:.3f}, \n 矩阵: \nTP:{TP}\nTN:{TN}\nFP:{FP}\nFN:{FN} \n"
                with open(otuput_file, 'a') as f:
                    f.write(output_result + '\n')
            
            with torch.no_grad():  
                model.eval()        
                valid_iterator = tqdm(valid_dataloader, desc=f"Evaluating Epoch {e}", unit="batch")
                all_valid_labels = []
                all_valid_preds = []
                for batch_idx, (image, label, table) in enumerate(valid_iterator):
                    image = image.to(device)
                    label = label.to(device)
                    table = table.to(device)
                    adj_pred, _, output, csdm_loss = model(image, table)
                    adj_true = compute_ground_truth_adjacency(label)
                    # print(output, label)
                    structure_loss, classification_loss_val = loss_fn(adj_pred, adj_true, _, label, output)
                    # print(csdm_loss, structure_loss, classification_loss_val)
                    # total_loss = 0.1 * csdm_loss + 0.1 * structure_loss + classification_loss_val
                    total_loss = classification_loss_val
                    total_valid_loss += total_loss.item()
                    
                    valid_metric.update(output, label)
                    # Collect labels and predictions for AUC calculation
                    all_valid_labels.extend(label.cpu().numpy())
                    all_valid_preds.extend(F.softmax(output, dim=1)[:, 1].cpu().detach().numpy())  # Probabilities for class 1

                valid_accuracy, valid_recall, valid_precision, valid_specificity = valid_metric.get_metrics()
                TP, TN, FP, FN = valid_metric.get_matrix()
                total_valid_loss /= len(valid_index)
                
                # Calculate AUC for validation
                valid_auc = roc_auc_score(all_valid_labels, all_valid_preds)
                    
            if e % print_freq == 0:
                output_result = f"Epoch [{e}/{epochs}]: \n val_loss={total_valid_loss:.3f}, \n val_accuracy={valid_accuracy}, \n val_recall={valid_recall}, \n val_precision={valid_precision}, \n val_specificity={valid_specificity}, \n val_AUC={valid_auc:.3f}, \n 矩阵: \nTP:{TP}\nTN:{TN}\nFP:{FP}\nFN:{FN} \n"
                with open(otuput_file, 'a') as f:
                    f.write(output_result + '\n')
            
            if valid_recall + valid_precision + valid_accuracy >= best_score:
                best_score = valid_recall + valid_precision + valid_accuracy
                best_epoch = e
                best_acc = valid_accuracy
                best_recall = valid_recall
                best_precision = valid_precision
                best_specificity = valid_specificity
                best_auc = valid_auc
                torch.save(model.state_dict(), best_model_path)  # Save best model

        print(best_score, best_epoch, best_acc)
        
        with open(otuput_file, 'a') as f:
            f.write(f"Fold {fold} Best Epoch: {best_epoch}, Best Acc: {best_acc}, Best Recall: {best_recall}, Best Prec: {best_precision}, Best Spec: {best_specificity}, Best AUC: {best_auc} \n")

        model.load_state_dict(torch.load(best_model_path))  # Load best model
        model.eval()
        test_metric = Metric_Manager(num_classes=1)
        all_test_labels = []
        all_test_preds = []
        with torch.no_grad():
            for image, label, table in test_dataloader:
                image = image.to(device)
                label = label.to(device)
                table = table.to(device)
                adj_pred, _, output, csdm_loss = model(image, table)

                test_metric.update(output, label)
                all_test_labels.extend(label.cpu().numpy())
                all_test_preds.extend(F.softmax(output, dim=1)[:, 1].cpu().detach().numpy())
        
        test_accuracy, test_recall, test_precision, test_specificity = test_metric.get_metrics()
        TP, TN, FP, FN = test_metric.get_matrix()
        test_auc = roc_auc_score(all_test_labels, all_test_preds)
        print(f"Test metrics for Fold {fold}: Accuracy={test_accuracy}, Recall={test_recall}, Precision={test_precision}, Specificity={test_specificity}, AUC={test_auc}")
        with open(otuput_file, 'a') as f:
            f.write(f"Fold {fold} Test Metrics: Accuracy={test_accuracy}, Recall={test_recall}, Precision={test_precision}, Specificity={test_specificity}, AUC={test_auc}\n 矩阵: \nTP:{TP}\nTN:{TN}\nFP:{FP}\nFN:{FN}")

    # end = time.time()
    # print(f"Cross-validation result:{epoch_acc_dict}")
    # print(f"Total training time: {(end - start) // 60}m {(end - start) % 60}s")


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    current_time = "{0:%Y%m%d_%H_%M}".format(datetime.now())
    args = parse_args()
    # 获取运行的设备
    if args.device != 'cpu':
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    dataset = infarction_Dataset(excel_file="1.xlsx",
                            img_prefix=args.data_dir)

    # 定义模型保存的文件夹
    model_dir = args.checkpoint_dir
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    # 训练的总轮数
    EPOCH = 200
    epoch = 0
    otuput_file = "main2.txt"
    # 设置随机数种子
    setup_seed(20)
    k_fold_cross_validation_with_test(device, dataset, EPOCH, k_fold=args.k_split_value,
                            batch_size=args.batch_size, workers=2, print_freq=1, model_dir=model_dir)