import sys
import os
import numpy as np
import pandas as pd
import random
import json
import csv
import datetime

from Client_logic.client_env_with_network_topology import read_txt_to_list_3, read_FoV_Trace, Environment
from Network_Trace.model.GraphAdjMatrix import GraphAdjMatrix, initialize_graph, mobility_control, dijkstra

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

RANDOM_SEED = 1

def get_topology(adjMatrix, node_list, time_ptr = 0): # every time_stamp has different topology

    for node in node_list:
        mobility_control(node, time_ptr)
    
    adjMatrix.get_new_AdjMatrix(node_list)
    
    ##------------ route decide algorithm -------------##
    # base on adjMatrix.propagation_matrix and adjMatrix.bandwidth_matrix
    start_node = 0
    end_node = len(adjMatrix.propagation_matrix) - 1
    path, propagation_delay = dijkstra(adjMatrix.propagation_matrix, start_node, end_node)
    
    real_bandwidth = float('inf')
    for i in range(len(path)-1):
        part_bandwidth = adjMatrix.bandwidth_matrix[i][i+1]
        real_bandwidth = real_bandwidth if real_bandwidth <= part_bandwidth else part_bandwidth
    
    ##------------ route decide algorithm -------------##

    return propagation_delay / 1000, real_bandwidth

if __name__ == "__main__":
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    dataset_name = "Fan_NOSSDAV_17"
    video_name = "coaster"
    user_name = "user21"

    # Network_Trace = read_txt_to_list_3(current_dir + "/Network_Trace/test/test_20") # 替换为topology
    # initialize graph
    node_list, ground_node_num, air_node_num, space_node_num = initialize_graph()
    adjMatrix = GraphAdjMatrix(node_list, ground_node_num, air_node_num, space_node_num)
    propagation_delay, bandwidth = get_topology(adjMatrix, node_list)
    Network_para = [propagation_delay, bandwidth]

    with open(current_dir + "/Video_Trace/Fan_NOSSDAV_17/coaster.json", 'r', encoding='utf-8') as file:
    # 直接使用json.load函数从文件对象中读取内容并转换为字典
        Video_Trace = json.load(file)  # 读出的key值是str，要转换
    FoV_trace = read_FoV_Trace(dataset_name, video_name, user_name)
    
    env = Environment(Network_para=Network_para, # Network_Trace is a number
                      video_trace=Video_Trace,
                      FoV_trace=FoV_trace,
                      M_WINDOW=5,
                      H_WINDOW=13,
                      FoV_model="pos_only",
                      use_true_saliency=False,
                      random_seed=RANDOM_SEED)
    
    output_file_path = current_dir + "/output/"+ dataset_name+"_"+video_name+"_"+user_name+"_"+formatted_time +"_client_test.csv"
    headers = ["chunk delay", "propagation delay", "real bandwidth", "real time", "req chunk id", "watching chunk id", "bandwidth ptr", "buffer size", "rebuffer time", "end of video"]
    
    # in the first_data, the rebuffer_time keep 0, but actually it should be the first_chunk_download_delay
    first_data = [env.first_chunk_download_delay, env.first_propagation_delay, env.network_trace, env.real_time, env.req_chunk_id, env.watching_chunk_id, env.bandwidth_ptr, env.buffer_size, env.rebuffer_time, False]
    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerow(first_data)

    step_count = 0
    data_list = []
    last_step_time = env.real_time
    end_of_video = False
    while True:
        step_count += 1 # base on step to decide Graph and Adjacent matrix
        
        time_ptr = env.real_time - last_step_time
        if end_of_video:
            env.reset()
        last_step_time = env.real_time


        propagation_delay, real_bandwidth = get_topology(adjMatrix, node_list, time_ptr)
        env.update_network_para(propagation_delay, real_bandwidth)

        chunk_delay, propagation_delay, real_time, req_chunk_id, watching_chunk_id, bandwidth_ptr, buffer_size, rebuffer_time, end_of_video = env.step()

        data = [chunk_delay, propagation_delay, real_bandwidth, real_time, req_chunk_id, watching_chunk_id, bandwidth_ptr, buffer_size, rebuffer_time, end_of_video]
        with open(output_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)

        if step_count > 300:
            break