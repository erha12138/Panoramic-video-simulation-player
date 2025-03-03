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
import tensorflow as tf
import a3c
import csv
import datetime
import abr_algorithms

RANDOM_SEED = 1
LINK_RTT = 80  # millisec
M_WINDOW=5
H_WINDOW=13
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE=0.0001
script_dir = os.path.dirname(os.path.abspath(__file__))
NN_MODEL = os.path.join(script_dir,'./models/pretrain_linear_reward.ckpt')

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

class A3Cbatch:
    def __init__(self):
        action_vec = np.zeros(A_DIM)

        self.s_batch = [np.zeros((S_INFO, S_LEN))]
        self.a_batch = [action_vec]
        self.r_batch = []
        self.entropy_record = []

    def get_s_batch(self):
        return self.s_batch
    def add(self,s,a,r,e):
        self.s_batch.append(s)
        self.a_batch.append(a)
        self.r_batch.append(r)
        self.entropy_record.append(e)

class Environment:  # 是给定了1、网络轨迹 2、视频轨迹 3、视野轨迹 的固定环境
    def __init__(
        self,
        Network_trace,   # 对应的实际带宽的轨迹
        video_trace,
        FoV_trace,
        buffer_len=30,
        M_WINDOW=5,
        H_WINDOW=13,
        model=None,
        use_true_saliency=False,
        random_seed=RANDOM_SEED,
        actor=None,
        a3cbatch=A3Cbatch(),
    ):
        
        self.network_trace = Network_trace
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
        self.model=model
        self.actor=actor
        self.CHUNK_LEN = 1.0
        self.M_WINDOW = M_WINDOW
        self.H_WINDOW = H_WINDOW
        self.a3cbatch=a3cbatch
        #记录上一次的运行环境
        self.last_step_info = dict()
        # randomize the start point of the trace
        # note: trace file starts with time 0 
        
        self.bandwidth_ptr = np.random.randint(1, int(len(self.video_trace)/2)) # 获取bandwidth的指针
        # test 
        # self.bandwidth_ptr = int(len(self.video_trace)) - 3 # 获取bandwidth的指针

        self.real_time = self.bandwidth_ptr # 实际执行时间
        self.watching_chunk_id = self.real_time # 用户实际观看到chunk_id，FoV依赖此chunk的id 
        self.req_chunk_id = self.watching_chunk_id # 请求的chunk的id，由请求序列决定，ABR依赖此chunk的id
        # 得额外处理第一块的情况，在初始化与reset时处理第一个chunk的执行逻辑
        # 初始状态没有FoV

        video_chunk_size_list = [self.video_trace[str(self.req_chunk_id)][str(tile)][str(0)] for tile in range(100)]
        video_chunk_size = sum(video_chunk_size_list) # 计算video_chunk_size
        self.first_chunk_download_delay, self.first_boardcasting_delay = self.first_download(video_chunk_size)

    def reset(self): # 重新观看一次这个视频，仍然是这个视频
        self.bandwidth_ptr = np.random.randint(1, int(len(self.video_trace)/2)) # 获取bandwidth的指针
        self.real_time = self.bandwidth_ptr # 实际执行时间
        self.watching_chunk_id = self.real_time # 用户实际观看到chunk_id，FoV依赖此chunk的id 
        self.req_chunk_id = self.watching_chunk_id # 请求的chunk的id，由请求序列决定，ABR依赖此chunk的id
 
    
    def first_download(self, video_chunk_size):
        first_download_time = 0
        broadcasting_delay = self.get_broadcasting_delay() 
        while True: # 下载第一个chunk
            if broadcasting_delay + (self.real_time - int(self.real_time)) + video_chunk_size / self.network_trace[self.bandwidth_ptr] > 1.0: # 进行下一轮循环
                video_chunk_size = video_chunk_size - self.network_trace[self.bandwidth_ptr] * 1 # 还要下载多少数据量
                self.bandwidth_ptr += 1
                self.real_time += 1 # 经历了1s
                # self.rebuffer_time += 1 # 第一个chunk必等待
                first_download_time += 1
            else:
                # 这个chunk下载完成，break
                download_time = video_chunk_size / self.network_trace[self.bandwidth_ptr]
                first_download_time += download_time
                self.real_time += download_time + broadcasting_delay # 不会超过1
                # watching_id不变，因为并没有被观看
                
                self.req_chunk_id += 1 # 待会请求下一个chunk了

                self.buffer_size += self.CHUNK_LEN # 直接加入第一个chunk
                # self.rebuffer_time += download_time # 第一个chunk必等待
                break
        return first_download_time, broadcasting_delay # 第一个chunk的下载时延与传播时延

    def download(self, video_chunk_size): 
        # watching_chunk_id 会随着 real_time 的值变化
        # rebuffer_time 要考虑
        real_download_time = 0
        end_of_video = False
        broadcasting_delay = self.get_broadcasting_delay()
        while True: # 下载一个chunk
            if broadcasting_delay + (self.real_time - int(self.real_time)) + video_chunk_size / self.network_trace[self.bandwidth_ptr] > 1.0: # 不止用一个bandwidth的值
                this_duration = 1 - (self.real_time - int(self.real_time))
                video_chunk_size = video_chunk_size - self.network_trace[self.bandwidth_ptr] * this_duration # 还要下载多少数据量
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
                download_time = video_chunk_size / self.network_trace[self.bandwidth_ptr]
                real_download_time += download_time
                self.real_time += (download_time + broadcasting_delay)
                # 不会超过1s，到这里相当于下载完成了
                self.req_chunk_id += 1 # 待会请求下一个chunk了
                # 防止broadcasting_delay太大导致的观看与请求错位，相当于会发生大量卡顿，并直接跳过一些chunk
                if self.watching_chunk_id >= self.req_chunk_id:
                    self.req_chunk_id = self.watching_chunk_id
                
                if self.buffer_size + self.CHUNK_LEN - download_time - broadcasting_delay > 0: # 说明没有发生rebuffer事件
                    self.buffer_size += self.CHUNK_LEN - download_time - broadcasting_delay
                else: # 发生rebuffer事件
                    self.rebuffer_time += download_time + broadcasting_delay - self.buffer_size - self.CHUNK_LEN
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
        return real_download_time, broadcasting_delay, end_of_video

    def get_broadcasting_delay(self):
        # 可能可以直接用一个文件，而拓扑的变化影响这个文件？？然后用self.real_time来索引
        return 0.08 # 先用80ms来测试
    
    def FoV_predict(self): # 要读实际的FoV轨迹
        model_name = self.FoV_model
        # 执行视野预测
        encoder_pos_inputs_for_sample = np.array([self.FoV_trace[self.watching_chunk_id - self.M_WINDOW:self.watching_chunk_id]])
        decoder_pos_inputs_for_sample = np.array([self.FoV_trace[self.watching_chunk_id:self.watching_chunk_id + 1]])
        if model_name == 'pos_only': 
            model_pred = self.model.predict([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_sample), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_sample)])[0]
            model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)

        print(model_prediction)     # model_pred 是欧拉的状态；model_prediction是笛卡尔的状态
        return model_pred, model_prediction # 返回视野预测结果，存在一定的预测窗口，由具体请求哪个chunk由self.req_chunk_id决定
    
    def get_next_video_chunk_size(self):
        req_trace=self.video_trace[str(self.req_chunk_id)]
        tile=req_trace[str(50)]

        next_vido_chunk_size=[]
        for i in range(A_DIM):
            next_vido_chunk_size.append(tile[str(i)])
        
        return next_vido_chunk_size

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


        abr_name="bb"
        if abr_name=="a3c":
            next_video_chunk_sizes=self.get_next_video_chunk_size()
            video_chunk_remain=len(self.video_trace)-self.req_chunk_id
            # self.last_step_info["next_video_chunk_sizes"] = next_video_chunk_sizes
            ground,focuse,state,action,reward,entroy=abr_algorithms.a3c_base(self.last_step_info,next_video_chunk_sizes,video_chunk_remain,actor,self.a3cbatch.get_s_batch())
            self.a3cbatch.add(state,action,reward,entroy)
        elif abr_name=="bb":
            ground,focuse=abr_algorithms.buffer_base(self.last_step_info)
        else:
            ground,focuse=abr_algorithms.normal()
        # 开始执行ABR
        # 暂时简单的ABR算法，focus tile 选最高质量，around tile 选次高质量；之后再修改
        tile_level = {}
        
        for tile_id in range(100):
            tile_level[tile_id] = self.video_trace[str(self.req_chunk_id)][str(tile_id)][str(0)]


        for tile_id in around_tile_id:
            tile_level[tile_id] = self.video_trace[str(self.req_chunk_id)][str(tile_id)][ground]
        tile_level[focus_tile_id] = self.video_trace[str(self.req_chunk_id)][str(focus_tile_id)][focuse]

        video_chunk_size = sum(tile_level.values()) # 计算video_chunk_size
        return tile_level, video_chunk_size # 返回每个tile的质量，是一个字典或列表；并计算video_chunk_size


    # 要读实际网络带宽轨迹 这个函数相当于一个step
    def step(self): # 直接给出下一个视频块的大小，具体计算在FoV与ABR结束后结合视野点给出

        delay = 0.0  # in s 总时延 = 下载时延 + 传播时延
        video_chunk_counter_sent = 0  # in bytes
        FoV_normalized_eulerian, FoV_cartesian = self.FoV_predict() # 执行预测视野，
        tile_level, video_chunk_size = self.ABR(FoV_normalized_eulerian)
        chunk_delay, broadcasting_delay, end_of_video = self.download(video_chunk_size)
        
        if end_of_video:
            self.reset() # step数量在外部控制
        


        return (chunk_delay,
                broadcasting_delay,
                self.real_time,
                self.req_chunk_id,
                self.watching_chunk_id,
                self.bandwidth_ptr,
                self.buffer_size,
                self.rebuffer_time,
                self.a3cbatch,
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
    
    
    output_file_path = "../output_dev/"+ dataset_name+"_"+video_name+"_"+user_name+"_"+formatted_time +"_client_test.csv"
    headers = ["chunk delay", "broadcasting delay", "real time", "req chunk id", "watching chunk id", "bandwidth ptr", "buffer size", "rebuffer time", "end of video"]
    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

    step_count = 0
    data_list = []

    
    with tf.compat.v1.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                state_dim=[S_INFO, S_LEN],
                                learning_rate=CRITIC_LR_RATE)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()  # save neural net parameters

        #加载actor模型和参数
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")
            # 提前加载fov模型
        data_set = "Fan_NOSSDAV_17"
        FoV_model="pos_only"
        fovmodel = create_pos_only_model(M_WINDOW, H_WINDOW)
        fovmodel.load_weights("../head-motion-prediction/" + data_set + "/" + FoV_model + "/Models_EncDec_eulerian_init_5_in_5_out_13_end_13" + '/weights.hdf5')

        env = Environment(Network_trace=Network_Trace,
                        video_trace=Video_Trace,
                        FoV_trace=FoV_trace,
                        M_WINDOW=5,
                        H_WINDOW=13,
                        actor=actor,
                        model=fovmodel,
                        use_true_saliency=False,
                        random_seed=RANDOM_SEED)

        while True:
            step_count += 1 # base on step to decide Graph and Adjacent matrix

            (chunk_delay, 
            broadcasting_delay, 
            real_time, 
            req_chunk_id, 
            watching_chunk_id, 
            bandwidth_ptr,
            buffer_size,
            rebuffer_time, 
            a3cbatch ,
            end_of_video) = env.step()
            
            # print("chunk delay:", chunk_delay, "real time:", real_time, "req chunk id:", req_chunk_id, "watching chunk id:", watching_chunk_id, "bandwidth ptr:", bandwidth_ptr, "buffer size:", buffer_size, "rebuffer time:", rebuffer_time)
            
            step_info = {
                "chunk delay": chunk_delay,
                "broadcasting delay": broadcasting_delay,
                "real time": real_time,
                "req chunk id": req_chunk_id,
                "watching chunk id": watching_chunk_id,
                "bandwidth ptr": bandwidth_ptr,
                "buffer size": buffer_size,
                "rebuffer time": rebuffer_time,
                "end of video": end_of_video,
                "a3cbatch":a3cbatch
            }

            env.last_step_info=step_info

            data = [chunk_delay, broadcasting_delay, real_time, req_chunk_id, watching_chunk_id, bandwidth_ptr, buffer_size, rebuffer_time, end_of_video]
            with open(output_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data)

            if step_count > 300:
                break