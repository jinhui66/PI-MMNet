import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch Cnn For Any Dataset')
    parser.add_argument('--data_dir', type=str, default="/data3/wangchangmiao/jinhui/DATA/infraction/processed_image", help='Path to dataset')
    parser.add_argument('--csv_file_path', type=str, default="./1.xlsx", help='Path to csv file')
    parser.add_argument('--Image_size', type=int, default=256, help='Size to reshape image')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Path to save model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of Epoch')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--k_split_value', type=int, default=5, help='k split value for k_fold mode')
    
    args = parser.parse_args()
    return args

# print(parse_args())
