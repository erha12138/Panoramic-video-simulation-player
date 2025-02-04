import sys
import os
import numpy as np
import pandas as pd
import random
import json
from position_only_baseline import create_pos_only_model
import csv
import datetime



if __name__ == "__main__":
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    dataset_name = "Fan_NOSSDAV_17"
    video_name = "coaster"
    user_name = "user21"

    Network_Trace = read_txt_to_list_3("../Network_Trace/test/test_20")
    with open("../Video_Trace/Fan_NOSSDAV_17/coaster.json", 'r', encoding='utf-8') as file:
    # 直接使用json.load函数从文件对象中读取内容并转换为字典
        Video_Trace = json.load(file)  # 读出的key值是str，要转换
    FoV_trace = read_FoV_Trace(dataset_name, video_name, user_name)
    
    env = Environment(Network_trace=Network_Trace,
                      video_trace=Video_Trace,
                      FoV_trace=FoV_trace,
                      M_WINDOW=5,
                      H_WINDOW=13,
                      FoV_model="pos_only",
                      use_true_saliency=False,
                      random_seed=RANDOM_SEED)
    
    output_file_path = "../output/"+ dataset_name+"_"+video_name+"_"+user_name+"_"+formatted_time +"_client_test.csv"
    headers = ["chunk delay", "broadcasting delay", "real time", "req chunk id", "watching chunk id", "bandwidth ptr", "buffer size", "rebuffer time", "end of video"]
    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

    step_count = 0
    data_list = []
    while True:
        step_count += 1 # base on step to decide Graph and Adjacent matrix

        

        chunk_delay, broadcasting_delay, real_time, req_chunk_id, watching_chunk_id, bandwidth_ptr, buffer_size, rebuffer_time, end_of_video = env.step()
        
        # print("chunk delay:", chunk_delay, "real time:", real_time, "req chunk id:", req_chunk_id, "watching chunk id:", watching_chunk_id, "bandwidth ptr:", bandwidth_ptr, "buffer size:", buffer_size, "rebuffer time:", rebuffer_time)
        data = [chunk_delay, broadcasting_delay, real_time, req_chunk_id, watching_chunk_id, bandwidth_ptr, buffer_size, rebuffer_time, end_of_video]
        with open(output_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)

        if step_count > 300:
            break