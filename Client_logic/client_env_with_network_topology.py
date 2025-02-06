import sys
import os
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
os.chdir(current_dir) # 切换到当前文件所在目录
sys.path.append('../head-motion-prediction/')
import numpy as np
import pandas as pd
import random
import json
from position_only_baseline import create_pos_only_model
import csv
import datetime


RANDOM_SEED = 1
LINK_RTT = 80  # millisec


# from head-motion-prediction/Utils.py
def cartesian_to_eulerian(x, y, z):
    r = np.sqrt(x*x+y*y+z*z)
    theta = np.arctan2(y, x)
    phi = np.arccos(z/r)
    # remainder is used to transform it in the positive range (0, 2*pi)
    theta = np.remainder(theta, 2*np.pi)
    return theta, phi

def eulerian_to_cartesian(theta, phi):
    x = np.cos(theta)*np.sin(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(phi)
    return np.array([x, y, z])

def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch):
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch] for batch in positions_in_batch]
    eulerian_batches = np.array(eulerian_batches) / np.array([2*np.pi, np.pi])
    return eulerian_batches

def transform_normalized_eulerian_to_cartesian(positions):
    positions = positions * np.array([2*np.pi, np.pi])
    eulerian_samples = [eulerian_to_cartesian(pos[0], pos[1]) for pos in positions]
    return np.array(eulerian_samples)

# 从笛卡尔坐标转换为tile id
def get_tile_id(FoV_array):
    [x,y] = list(FoV_array)

    x_index = int(x * 9)
    y_index = int(y * 9)
    tile_id = x_index + y_index * 10
    return tile_id

def get_around_tile_id(focus_tile_id):
    around_tile_id = []
    x = focus_tile_id % 10
    y = focus_tile_id // 10
    for i in range(-1, 2):
        for j in range(-1, 2):
            new_x = (x + i) % 10
            new_y = (y + j) % 10
            around_tile_id.append(new_x + new_y * 10)
    return around_tile_id

class Environment:  # 是给定了1、网络轨迹 2、视频轨迹 3、视野轨迹 的固定环境
    def __init__(
        self,
        Network_para,
        video_trace,
        FoV_trace,
        buffer_len=30,
        M_WINDOW=5,
        H_WINDOW=13,
        data_set = "Fan_NOSSDAV_17",
        FoV_model="pos_only",
        use_true_saliency=False,
        random_seed=RANDOM_SEED,
    ):
        
        self.propagation_delay, self.network_trace  = Network_para # [propagation_delay, bandwidth]
        np.random.seed(random_seed)
        self.video_trace = video_trace
        self.FoV_trace = FoV_trace # 获取某用户的视野轨迹
        self.FoV_model = FoV_model
        self.use_true_saliency = use_true_saliency
        self.video_chunk_counter = 0   # 第几个CHUNK
        self.buffer_size = 0
        self.rebuffer_time = 0
        self.data_set = data_set
        self.buffer_len = buffer_len # 
        
        self.CHUNK_LEN = 1.0
        self.M_WINDOW = M_WINDOW
        self.H_WINDOW = H_WINDOW

        # randomize the start point of the trace
        # note: trace file starts with time 0 
        
        self.bandwidth_ptr = np.random.randint(1, int(len(self.video_trace)/2)) # 获取bandwidth的指针

        self.real_time = self.bandwidth_ptr # 实际执行时间
        self.watching_chunk_id = self.real_time # 用户实际观看到chunk_id，FoV依赖此chunk的id 
        self.req_chunk_id = self.watching_chunk_id # 请求的chunk的id，由请求序列决定，ABR依赖此chunk的id
        # 得额外处理第一块的情况，在初始化与reset时处理第一个chunk的执行逻辑
        # 初始状态没有FoV

        video_chunk_size_list = [self.video_trace[str(self.req_chunk_id)][str(tile)][str(0)] for tile in range(100)]
        video_chunk_size = sum(video_chunk_size_list) # 计算video_chunk_size
        self.first_chunk_download_delay, self.first_propagation_delay = self.first_download(video_chunk_size)

    def reset(self): # 重新观看一次这个视频，仍然是这个视频
        self.bandwidth_ptr = np.random.randint(1, int(len(self.video_trace)/2)) # 获取bandwidth的指针
        # self.network_trace = Network_Trace
        self.real_time = self.bandwidth_ptr # 实际执行时间
        self.watching_chunk_id = self.real_time # 用户实际观看到chunk_id，FoV依赖此chunk的id 
        self.req_chunk_id = self.watching_chunk_id # 请求的chunk的id，由请求序列决定，ABR依赖此chunk的id
    
    def reset_network_trace(self, Network_Trace):
        self.network_trace = Network_Trace
    
    def first_download(self, video_chunk_size):
        first_download_time = 0
        while True: # 下载第一个chunk
            if self.propagation_delay + (self.real_time - int(self.real_time)) + video_chunk_size / self.network_trace > 1.0: # 进行下一轮循环
                video_chunk_size = video_chunk_size - self.network_trace * 1 # 还要下载多少数据量
                self.bandwidth_ptr += 1
                self.real_time += 1 # 经历了1s
                # self.rebuffer_time += 1 # 第一个chunk必等待
                first_download_time += 1
            else:
                # 这个chunk下载完成，break
                download_time = video_chunk_size / self.network_trace
                first_download_time += download_time
                self.real_time += download_time + self.propagation_delay # 不会超过1
                # watching_id不变，因为并没有被观看
                
                self.req_chunk_id += 1 # 待会请求下一个chunk了

                self.buffer_size += self.CHUNK_LEN # 直接加入第一个chunk
                # self.rebuffer_time += download_time # 第一个chunk必等待
                break
        return first_download_time, self.propagation_delay # 第一个chunk的下载时延与传播时延

    def download(self, video_chunk_size): 
        # watching_chunk_id 会随着 real_time 的值变化
        # rebuffer_time 要考虑
        real_download_time = 0
        end_of_video = False
        while True: # 下载一个chunk
            if self.propagation_delay + (self.real_time - int(self.real_time)) + video_chunk_size / self.network_trace > 1.0: # 不止用一个bandwidth的值
                this_duration = 1 - (self.real_time - int(self.real_time))
                video_chunk_size = video_chunk_size - self.network_trace * this_duration # 还要下载多少数据量
                real_download_time += this_duration
                # self.bandwidth_ptr += 1
                self.watching_chunk_id += 1 # 该看下一个chunk了
                self.bandwidth_ptr += 1
                self.real_time = int(self.real_time) + 1 # 到了下一个1s
                
                if self.buffer_size - this_duration >= 0: # 说明没发生rebuffer事件
                    self.buffer_size -= this_duration # 下载了this_durations 的时长，但没下载完一整个chunk，buffer_size得减1s了
                else:
                    # 发生rebuffer事件
                    self.rebuffer_time += this_duration - self.buffer_size
                    self.buffer_size = 0
            else:
                # 这个chunk下载完成，break
                download_time = video_chunk_size / self.network_trace
                real_download_time += download_time
                self.real_time += (download_time + self.propagation_delay)
                # 不会超过1s，到这里相当于下载完成了
                self.req_chunk_id += 1 # 待会请求下一个chunk了
                # 防止self.propagation_delay太大导致的观看与请求错位，相当于会发生大量卡顿，并直接跳过一些chunk
                if self.watching_chunk_id >= self.req_chunk_id:
                    self.req_chunk_id = self.watching_chunk_id
                
                if self.buffer_size + self.CHUNK_LEN - download_time - self.propagation_delay > 0: # 说明没有发生rebuffer事件
                    self.buffer_size += self.CHUNK_LEN - download_time - self.propagation_delay
                else: # 发生rebuffer事件
                    self.rebuffer_time += download_time + self.propagation_delay - self.buffer_size - self.CHUNK_LEN
                    self.buffer_size = 0

                # 如果buffer满了，那就等此chunk下载完，而不再请求新的chunk
                if self.buffer_size >= self.buffer_len:
                    waiting_time = self.buffer_size - self.buffer_len
                    self.real_time += waiting_time
                    self.bandwidth_ptr = int(self.real_time)
                    self.watching_chunk_id = int(self.real_time)
                    self.buffer_size = self.buffer_len

                if self.req_chunk_id >= len(self.video_trace): # 如果已经请求超过最后一个chunk了
                    end_of_video = True # 请求视频结束了，需要重新初始化
                break      
        return real_download_time, self.propagation_delay, end_of_video

    def update_network_para(self, propagation_delay, bandwidth): # propagation delay and bandwidth
        
        self.propagation_delay = propagation_delay # in s
        self.network_trace = bandwidth
        
        # return propagation_delay, bandwidth 
    
    def FoV_predict(self): # 要读实际的FoV轨迹
        model_name = self.FoV_model
        if model_name == 'TRACK':
            if self.use_true_saliency:
                model = create_TRACK_model(self.M_WINDOW, self.H_WINDOW, NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL)
            else:
                model = create_TRACK_model(self.M_WINDOW, self.H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
        elif model_name == 'TRACK_AblatSal':
            if self.use_true_saliency:
                model = create_TRACK_AblatSal_model(self.M_WINDOW, self.H_WINDOW, NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL)
            else:
                model = create_TRACK_AblatSal_model(self.M_WINDOW, self.H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
        elif model_name == 'TRACK_AblatFuse':
            if self.use_true_saliency:
                model = create_TRACK_AblatFuse_model(self.M_WINDOW, self.H_WINDOW, NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL)
            else:
                model = create_TRACK_AblatFuse_model(self.M_WINDOW, self.H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
        elif model_name == 'CVPR18':
            if self.use_true_saliency:
                model = create_CVPR18_model(self.M_WINDOW, self.H_WINDOW, NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL)
            else:
                model = create_CVPR18_model(self.M_WINDOW, self.H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
        elif model_name == 'MM18':
            mm18_models = []
            for _self.H_WINDOW in range(self.H_WINDOW):
                mm18_models.append(MM18_model.create_MM18_model())
        elif model_name == 'pos_only':
            model = create_pos_only_model(self.M_WINDOW, self.H_WINDOW)
        elif model_name == 'pos_only_3d_loss':
            obj = Pos_Only_Class(self.H_WINDOW)
            model = obj.get_model()
        elif model_name == 'CVPR18_orig':
            if self.use_true_saliency:
                model = create_CVPR18_orig_Model(self.M_WINDOW, NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL)
            else:
                model = create_CVPR18_orig_Model(self.M_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
        
        # if model_name not in ['no_motion', 'most_salient_point', 'true_saliency', 'content_based_saliency', 'MM18']:
        #     model.summary()
        # elif model_name == 'MM18':
        #     mm18_models[0].summary()

        if self.watching_chunk_id < self.M_WINDOW: # 如果还没有足够的数据
            return None # 返回空
        else:
            if model_name not in ['pos_only', 'no_motion', 'true_saliency', 'content_based_saliency', 'pos_only_3d_loss', 'MM18']:
                encoder_sal_inputs_for_sample = np.array([np.expand_dims(all_saliencies[video][x_i - M_WINDOW + 1:x_i + 1], axis=-1)])
                decoder_sal_inputs_for_sample = np.array([np.expand_dims(all_saliencies[video][x_i + 1:x_i + H_WINDOW + 1], axis=-1)])
            if model_name in ['true_saliency', 'content_based_saliency']:
                decoder_true_sal_inputs_for_sample = most_salient_points_per_video[video][x_i + 1:x_i + H_WINDOW + 1]
            if model_name == 'CVPR18_orig':
                # ToDo when is CVPR18_orig the input is the concatenation of encoder_pos_inputs and decoder_pos_inputs
                encoder_pos_inputs_for_sample = np.array([all_traces[video][user][x_i - M_WINDOW + 1:x_i + 1]])
            elif model_name == 'MM18':
                encoder_sal_inputs_for_sample = np.array([np.concatenate((all_saliencies[video][x_i-M_WINDOW+1:x_i+1], all_headmaps[video][user][x_i-M_WINDOW+1:x_i+1]), axis=1)])
            else:
                encoder_pos_inputs_for_sample = np.array([self.FoV_trace[self.watching_chunk_id - self.M_WINDOW:self.watching_chunk_id]])
                decoder_pos_inputs_for_sample = np.array([self.FoV_trace[self.watching_chunk_id:self.watching_chunk_id + 1]])
        # 读入模型权重
        if model_name not in ['no_motion', 'most_salient_point', 'true_saliency', 'content_based_saliency', 'MM18']:
            model.load_weights("../head-motion-prediction/" + self.data_set + "/" + self.FoV_model + "/Models_EncDec_eulerian_init_5_in_5_out_13_end_13" + '/weights.hdf5')
        # 执行视野预测
        if model_name == 'pos_only': 
            model_pred = model.predict([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_sample), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_sample)])[0]
            model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)

        print(model_prediction)     # model_pred 是欧拉的状态；model_prediction是笛卡尔的状态
        return model_pred, model_prediction # 返回视野预测结果，存在一定的预测窗口，由具体请求哪个chunk由self.req_chunk_id决定

    def ABR(self, FoV_predict_eulerian): # chunk_id 是要请求的下一个chunk的id 
        # 给出实际ABR算法，计算出tile_quality = 【tile1：1，tile2：1,....】
        # FoV_predict[] # 从 0 开始，0代表下一个
        # self.watching_chunk_id # 现在在观看这个
        # self.req_chunk_id     # 要请求这个，则要找到FoV中指向此chunk_id的部分
        FoV_duration = self.req_chunk_id - self.watching_chunk_id # 使用了预测多远的未来
        if FoV_duration >= len(FoV_predict_eulerian):
            FoV_duration = len(FoV_predict_eulerian) # 防止越界，但FoV肯定就没用了
        focus_tile_id = get_tile_id(FoV_predict_eulerian[FoV_duration - 1]) # 用于ABR的实际视野
        around_tile_id = get_around_tile_id(focus_tile_id)

        # 开始执行ABR
        # 暂时简单的ABR算法，focus tile 选最高质量，around tile 选次高质量；之后再修改
        tile_level = {}
        
        for tile_id in range(100):
            tile_level[tile_id] = self.video_trace[str(self.req_chunk_id)][str(tile_id)][str(0)]
# 在最低级的tile之后再赋值高级的，这样高级会覆盖低级
        for tile_id in around_tile_id:
            tile_level[tile_id] = self.video_trace[str(self.req_chunk_id)][str(tile_id)][str(5)]
# 在around之后再赋值focus，这样focus会覆盖around
        tile_level[focus_tile_id] = self.video_trace[str(self.req_chunk_id)][str(focus_tile_id)][str(7)]

        video_chunk_size = sum(tile_level.values()) # 计算video_chunk_size
        return tile_level, video_chunk_size # 返回每个tile的质量，是一个字典或列表；并计算video_chunk_size


    # 要读实际网络带宽轨迹 这个函数相当于一个step
    def step(self): # 直接给出下一个视频块的大小，具体计算在FoV与ABR结束后结合视野点给出

        delay = 0.0  # in s 总时延 = 下载时延 + 传播时延
        video_chunk_counter_sent = 0  # in bytes
        FoV_normalized_eulerian, FoV_cartesian = self.FoV_predict() # 执行预测视野，
        tile_level, video_chunk_size = self.ABR(FoV_normalized_eulerian)
        chunk_delay, propagation_delay, end_of_video = self.download(video_chunk_size)
        
        # if end_of_video:
        #     self.reset() # step数量在外部控制
        
        return (chunk_delay,
                propagation_delay,
                self.real_time,
                self.req_chunk_id,
                self.watching_chunk_id,
                self.bandwidth_ptr,
                self.buffer_size,
                self.rebuffer_time,
                end_of_video)
    

def read_txt_to_list_3(file_path):
    # 使用numpy的genfromtxt函数读取文件，设置dtype为合适的数据类型（比如float），按行读取
    lines = np.genfromtxt(file_path, dtype=float, delimiter=None, encoding='utf-8')
    return lines.tolist()

import json
import os
import pandas as pd
def read_FoV_Trace(dataset_name, video, user):
    dataset_path = "../head-motion-prediction/"+dataset_name+"/sampled_dataset/"
    
    path = os.path.join(dataset_path, video, user)
    data = pd.read_csv(path, header=None)
    return data.values[:, 1:]


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

        

        chunk_delay, propagation_delay, real_time, req_chunk_id, watching_chunk_id, bandwidth_ptr, buffer_size, rebuffer_time, end_of_video = env.step()
        
        # print("chunk delay:", chunk_delay, "real time:", real_time, "req chunk id:", req_chunk_id, "watching chunk id:", watching_chunk_id, "bandwidth ptr:", bandwidth_ptr, "buffer size:", buffer_size, "rebuffer time:", rebuffer_time)
        data = [chunk_delay, propagation_delay, real_time, req_chunk_id, watching_chunk_id, bandwidth_ptr, buffer_size, rebuffer_time, end_of_video]
        with open(output_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)

        if step_count > 300:
            break