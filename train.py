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

def main(args):

    try:
        with open(args.config_filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        data_args = config['data']
        model_args = config['model']
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("INFO: device = ", device)
        
        model = STAG_GCN(node_num=data_args['node_num'], seq_len=model_args['his_num'], pred_len=model_args['pred_num'], graph_dim=model_args['graph_dim'], tcn_dim=model_args['tcn_dim'], atten_head=model_args['atten_head'], choice=model_args['choice']).to(device)
        print(f"Model params: graph_dim = {model_args['graph_dim']}, tcn_dim={model_args['tcn_dim']}, atten_head = {model_args['atten_head']}")
        print('INFO: Model parameters_count:', count_parameters(model))

        model_optimizer = torch.optim.Adam(model.parameters(), lr = model_args['base_lr'], weight_decay = 5e-4, eps=1e-4)
        # model_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[5, 20, 50], gamma=0.5)
        # criterion = torch.nn.MSELoss()
        criterion = torch.nn.L1Loss()

        train_Dataset = geometric_dataset(dataset_path = data_args['dataset_path'],\
                                        adjacency_matrix_path = data_args['adjacency_matrix_path'],\
                                        dtw_matrix_path = data_args['dtw_matrix_path'],\
                                        node_num = data_args['node_num'],\
                                        speed_mean = data_args['speed_mean'],\
                                        speed_std = data_args['speed_std'],\
                                        his_num = model_args['his_num'], pred_num = model_args['pred_num'],\
                                        split_point_start = 0, split_point_end= int(data_args['length'] * 0.7 * 144), type='Train')
        train_dataloader = DataLoader(train_Dataset, batch_size = data_args['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

        val_Dataset = geometric_dataset(dataset_path = data_args['dataset_path'],\
                                        adjacency_matrix_path = data_args['adjacency_matrix_path'],\
                                        dtw_matrix_path = data_args['dtw_matrix_path'],\
                                        node_num = data_args['node_num'],\
                                        speed_mean = data_args['speed_mean'],\
                                        speed_std = data_args['speed_std'],\
                                        his_num = model_args['his_num'], pred_num = model_args['pred_num'],\
                                        split_point_start = int(data_args['length'] * 0.7 * 144), split_point_end= int(data_args['length'] * 0.8 * 144), type='Validation')
        val_dataloader = DataLoader(val_Dataset, batch_size = data_args['batch_size'], num_workers=0, pin_memory=True)

        print("INFO: Dataloader finish.")
        epochs = model_args['epochs']
        result_record = {}
        result_record['MSE'], result_record['RMSE'], result_record['MAE'], result_record['MAPE'] = np.array([100,100,100]), np.array([100,100,100]), np.array([100,100,100]), np.array([100,100,100])

        for epoch in range(epochs):
            model.train()        
            start = time.time()
            total_loss = 0
            for step, data in enumerate(train_dataloader):
                edge_index, dtw_edge_index = data.edge_index.to(device), None
                x_data = data.x.to(device)
                y_data = data.y.to(device)
                predictions = model(x_data, edge_index, dtw_edge_index)
                predictions = torch.reshape(predictions, (-1, data_args['node_num'], model_args['pred_num']))
                ground_truth = torch.reshape(y_data, (-1, data_args['node_num'], model_args['pred_num']))
                loss = criterion(predictions, ground_truth)
                model_optimizer.zero_grad()
                loss.backward()
                if model_args['max_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=model_args['max_grad_norm']
                    )
                model_optimizer.step()
                total_loss = total_loss + loss.item()
            total_loss = total_loss / (step + 1)
            end = time.time()
            lr_scheduler.step()
            print(f"INFO: Epoch {epoch}/{epochs}: train loss = {total_loss}  training time = {end - start}")

            if(epoch % 10 == 0):
                model.eval()
                start = time.time()
                prediction_result, ground_truth_result = [], []
                with torch.no_grad():
                    for step_test, data in enumerate(val_dataloader):
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
                print(f"Validation time: {end - start}")

                prediction_result = prediction_result.cpu().numpy()
                ground_truth_result = ground_truth_result.cpu().numpy()
                result = metric_func(prediction_result, ground_truth_result, times=6)
                total_MSE, total_RMSE, total_MAE, total_MAPE = result['MSE'], result['RMSE'], result['MAE'], result['MAPE']
                
                if (result_record['RMSE'][0] > total_RMSE[0]):
                    result_record['MSE'], result_record['RMSE'], result_record['MAE'], result_record['MAPE'] = total_MSE, total_RMSE, total_MAE, total_MAPE
                    print("---------------------------------------")
                    print("========== New record result ==========")
                    print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
                    print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
                    print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
                    print(f"[Config] name:{data_args['name']}, choice:{model_args['choice']}, graph_dim:{model_args['graph_dim']}")
                    print("---------------------------------------")
                    torch.save(model.state_dict(), model_args['model_filename'])
                    np.save(model_args['prediction_filename'], prediction_result)
                    np.save(model_args['ground_truth_filename'], ground_truth_result)
                    print("INFO: Save model ...")
                else:
                    print("---------------------------------------")
                    print("========== Evaluate results ==========")
                    print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
                    print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
                    print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
                    print(f"[Config] name:{data_args['name']}, choice:{model_args['choice']}, graph_dim:{model_args['graph_dim']}")
                    print("---------------------------------------")
        
        # Best record 
        MSE, RMSE, MAE, MAPE = result_record['MSE'], result_record['RMSE'], result_record['MAE'], result_record['MAPE']
        print("---------------------------------------")
        print("========= Best record results =========")
        print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(MAE[0], MAE[1], MAE[2], MAE[3], MAE[4], MAE[5]))
        print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(MAPE[0] * 100, MAPE[1] * 100, MAPE[2] * 100, MAPE[3] * 100, MAPE[4] * 100, MAPE[5] * 100))
        print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(RMSE[0], RMSE[1], RMSE[2], RMSE[3], RMSE[4], RMSE[5]))
        print(f"[Config] name:{data_args['name']}, choice:{model_args['choice']}, graph_dim:{model_args['graph_dim']}")
        print("---------------------------------------")
    
    except KeyboardInterrupt:
        MSE, RMSE, MAE, MAPE = result_record['MSE'], result_record['RMSE'], result_record['MAE'], result_record['MAPE']
        print("---------------------------------------")
        print("========= Best record results =========")
        print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(MAE[0], MAE[1], MAE[2], MAE[3], MAE[4], MAE[5]))
        print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(MAPE[0] * 100, MAPE[1] * 100, MAPE[2] * 100, MAPE[3] * 100, MAPE[4] * 100, MAPE[5] * 100))
        print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(RMSE[0], RMSE[1], RMSE[2], RMSE[3], RMSE[4], RMSE[5]))
        print("---------------------------------------")
        print(" MAE: %.3f/ %.3f/ %.3f"%(MAE[0], MAE[2], MAE[5]))
        print("MAPE: %.3f/ %.3f/ %.3f"%(MAPE[0] * 100, MAPE[2] * 100, MAPE[5] * 100))
        print("RMSE: %.3f/ %.3f/ %.3f"%(RMSE[0], RMSE[2], RMSE[5]))
        print(f"[Config] name:{data_args['name']}, choice:{model_args['choice']}, graph_dim:{model_args['graph_dim']}")
        print("---------------------------------------")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    args = parser.parse_args()
    main(args)
