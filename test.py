import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import numpy as np
import math

import yaml
import torch
import argparse
import random
from geometric_dataset import geometric_dataset
from torch_geometric.data import Data, Dataset, DataLoader
from utils import *
from Models import STAG_GCN
import warnings 
warnings.filterwarnings('ignore')

def main(args):
        with open(args.config_filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        data_args = config['data']
        model_args = config['model']
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        model = STAG_GCN(node_num=data_args['node_num'], seq_len=model_args['his_num'], pred_len=model_args['pred_num'], graph_dim=model_args['graph_dim'], tcn_dim=model_args['tcn_dim'], atten_head=model_args['atten_head'], choice=model_args['choice']).to(device)
        model.load_state_dict(torch.load(model_args['model_filename'], map_location=device))
        print('INFO: Load model successful.')

        test_Dataset = geometric_dataset(dataset_path = data_args['dataset_path'],\
                                        adjacency_matrix_path = data_args['adjacency_matrix_path'],\
                                        dtw_matrix_path = data_args['dtw_matrix_path'],\
                                        node_num = data_args['node_num'],\
                                        speed_mean = data_args['speed_mean'],\
                                        speed_std = data_args['speed_std'],\
                                        his_num = model_args['his_num'], pred_num = model_args['pred_num'],\
                                        split_point_start = int(data_args['length'] * 0.8 * 144), split_point_end= int(data_args['length'] * 144), type='Test')
        test_dataloader = DataLoader(test_Dataset, batch_size = data_args['batch_size'], num_workers=0, pin_memory=True)

        print("INFO: Dataloader finish.")
        epochs = model_args['epochs']
        result_record = {}
        result_record['MSE'], result_record['RMSE'], result_record['MAE'], result_record['MAPE'] = np.array([100,100,100]), np.array([100,100,100]), np.array([100,100,100]), np.array([100,100,100])

        model.eval()
        start = time.time()
        prediction_result, ground_truth_result = [], []
        with torch.no_grad():
            for step_test, data in enumerate(test_dataloader):
                edge_index, dtw_edge_index = data.edge_index.to(device), None
                x_data = data.x.to(device)
                y_data = data.y.to(device)
                predictions = model(x_data, edge_index, dtw_edge_index)
                predictions = torch.reshape(predictions, (-1, data_args['node_num'], model_args['pred_num']))
                ground_truth = torch.reshape(y_data, (-1, data_args['node_num'], model_args['pred_num']))
                pred_ = predictions.permute(0, 2, 1)
                y_ = ground_truth.permute(0, 2, 1)
                prediction_result.append(pred_)
                ground_truth_result.append(y_)
            prediction_result = torch.cat(prediction_result, dim=0)
            ground_truth_result = torch.cat(ground_truth_result, dim=0)
        end = time.time()
        print(f"Testing time: {end - start}")

        prediction_result = prediction_result.cpu().numpy()
        ground_truth_result = ground_truth_result.cpu().numpy()
        result = metric_func(prediction_result, ground_truth_result, times=6)
        total_MSE, total_RMSE, total_MAE, total_MAPE = result['MSE'], result['RMSE'], result['MAE'], result['MAPE']

        print("========== Evaluate results ==========")
        print(f"未来10分钟的预测准确度: {100-total_MAPE[0]*100:.2f}%")
        print(f"未来30分钟的预测准确度: {100-total_MAPE[2]*100:.2f}%")
        print(f"未来60分钟的预测准确度: {100-total_MAPE[5]*100:.2f}%")
        print("---------------------------------------")
 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    args = parser.parse_args()
    main(args)
