import sys
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
os.chdir(current_dir) # 切换到当前文件所在目录
sys.path.append('../head-motion-prediction/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from position_only_baseline import create_pos_only_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import a3c
import csv
import datetime
import abr_algorithms
import env
import load_trace


RANDOM_SEED = 1
B_IN_MB=1000000
LINK_RTT = 80  # millisec
M_WINDOW=5
H_WINDOW=13
VIDEO_BIT_RATE =[300,720,1080,1260,1440,2160,3120,4080]
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 8
# 定义 QoE 计算所需的权重
BITRATE_WEIGHT = 6.25
BITRATE_CHANGE_WEIGHT = -0.5
REBUFFER_WEIGHT = -2.0

ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE=0.0001
script_dir = os.path.dirname(os.path.abspath(__file__))
A3C_MODEL = os.path.join(script_dir,'./rl_models/pensieve/model/test_model_2800.ckpt')

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
    def get_r_batch(self):
        return self.r_batch
    def get_a_batch(self):
        return self.a_batch
    def add(self,s,a,r,e):
        self.s_batch.append(s)
        self.a_batch.append(a)
        self.r_batch.append(r)
        self.entropy_record.append(e)

class Environment:  # 是给定了1、网络轨迹 2、视频轨迹 3、视野轨迹 的固定环境
    def __init__(
        self,
        all_cooked_time,
        all_cooked_bw,   # 对应的实际带宽的轨迹
        video_trace,
        FoV_trace,
        buffer_len=30,
        M_WINDOW=5,
        H_WINDOW=13,
        ABR_algorithm=None,
        model=None,
        use_true_saliency=False,
        random_seed=RANDOM_SEED,
        actor=None,
        a3cbatch=A3Cbatch(),
    ):
        
        self.net_env=env.Environment(all_cooked_time,all_cooked_bw,video_size_file='./video_sizes/video_size_')
        self.all_cooked_bw=all_cooked_bw,
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
        self.ABR_algorithm=ABR_algorithm
        self.a3cbatch=a3cbatch
        #记录上一次的运行环境
        self.last_step_info = dict()
        self.bitrate_history = []
        self.rebuffer_history = []
        self.bandwidth_ptr = np.random.randint(1, int(len(self.video_trace)/2)) # 获取bandwidth的指针/
        # self.bandwidth_ptr=2
        self.base_time=self.bandwidth_ptr
        self.real_time = self.bandwidth_ptr # 实际执行时间
        self.watching_chunk=self.real_time
        self.watching_chunk_id = self.real_time # 用户实际观看到chunk_id，FoV依赖此chunk的id 
        self.req_chunk_id = self.watching_chunk_id # 请求的chunk的id，由请求序列决定，ABR依赖此chunk的id
        # 得额外处理第一块的情况，在初始化与reset时处理第一个chunk的执行逻辑
        # 初始状态没有FoV

        # video_chunk_size_list = [self.video_trace[str(self.req_chunk_id)][str(tile)][str(0)] for tile in range(100)]
        self.last_step_info["bit rate"]=str(5)
        self.last_step_info["video chunk size"]=self.get_next_video_chunk_size()[5]
        self.last_step_info["buffer size"]=1
        self.last_step_info["delay"]=1000
        self.last_step_info["rebuffer time"]=0

        # video_chunk_size = sum(video_chunk_size_list)# 计算video_chunk_size  单位 B
        # self.first_chunk_download_delay, self.first_boardcasting_delay = self.first_download(video_chunk_size)

    def reset(self): # 重新观看一次这个视频，仍然是这个视频
        self.bandwidth_ptr = np.random.randint(1, int(len(self.video_trace)/2)) # 获取bandwidth的指针
        self.base_time=self.bandwidth_ptr
        self.real_time = self.bandwidth_ptr # 实际执行时间
        self.watching_chunk=self.real_time
        self.watching_chunk_id = self.real_time # 用户实际观看到chunk_id，FoV依赖此chunk的id 
        self.req_chunk_id = self.watching_chunk_id+int(self.buffer_size) # 请求的chunk的id，由请求序列决定，ABR依赖此chunk的id
 
    def get_qoe_metrics(self):
        # 计算当前视频比特率得分
        current_bitrate = int(self.last_step_info["bit rate"])
        bitrate_score=50+current_bitrate*BITRATE_WEIGHT
        # 计算最近 20 次比特率变换频率和幅度得分
        if len(self.bitrate_history) < 20:
            bitrate_changes = self.bitrate_history
        else:
            bitrate_changes = self.bitrate_history[-20:]

        bitrate_change_score = 0
        if len(bitrate_changes) > 1:
            for i in range(1, len(bitrate_changes)):
                # 计算bitrate_changes的均值
                mean_bitrate_changes = sum(bitrate_changes) / len(bitrate_changes)
                # 计算bitrate_changes的均方差
                mse_bitrate_changes = sum((x - mean_bitrate_changes) ** 2 for x in bitrate_changes) / len(bitrate_changes)
            bitrate_change_score = BITRATE_CHANGE_WEIGHT * mse_bitrate_changes

        # 计算 rebuffer 事件得分
        recent_rebuffers = sum(self.rebuffer_history[-20:]) if self.rebuffer_history else 0
        rebuffer_score = REBUFFER_WEIGHT * recent_rebuffers

        # 计算总的 QoE 指数
        qoe_index = bitrate_score + bitrate_change_score + rebuffer_score

        return qoe_index
    


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

        # print(model_prediction)     # model_pred 是欧拉的状态；model_prediction是笛卡尔的状态
        return model_pred, model_prediction # 返回视野预测结果，存在一定的预测窗口，由具体请求哪个chunk由self.req_chunk_id决定
    
    def get_next_video_chunk_size(self):
        req_trace=self.video_trace[str(self.req_chunk_id)]
        
        next_vido_chunk_size=np.zeros(A_DIM)
        for i in range(A_DIM):
            for tile_id in range(len(req_trace)):
                tile=req_trace[str(tile_id)]
                next_vido_chunk_size[i]+=tile[str(i)]*B_IN_MB
        
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


        a_batch=self.a3cbatch.get_a_batch()
        if len(a_batch) >= 15:
            std_bit = np.std(a_batch[-15:])
        else:
            std_bit=0

        abr_name=self.ABR_algorithm
        if abr_name=="pensieve":
            next_video_chunk_sizes=self.get_next_video_chunk_size()
            video_chunk_remain=len(self.video_trace)-self.req_chunk_id
            # self.last_step_info["next_video_chunk_sizes"] = next_video_chunk_sizes
            around,focuse,state,action,reward,entroy=abr_algorithms.a3c_base(self.last_step_info,next_video_chunk_sizes,video_chunk_remain,actor,self.a3cbatch.get_s_batch(),std_bit)
            self.a3cbatch.add(state,action,reward,entroy)
        elif abr_name=="bb":
            around,focuse=abr_algorithms.buffer_base(self.last_step_info)
        elif abr_name=="mpc":
            next_video_chunk_sizes=self.get_next_video_chunk_size()
            video_chunk_remain=len(self.video_trace)-self.req_chunk_id
            # self.last_step_info["next_video_chunk_sizes"] = next_video_chunk_sizes
            around,focuse,state,action=abr_algorithms.mpc_base(self.last_step_info,next_video_chunk_sizes,video_chunk_remain,self.a3cbatch.get_s_batch(),std_bit)
            reward=0
            entroy=0
            self.a3cbatch.add(state,action,reward,entroy)
        elif abr_name=="dqn":
            next_video_chunk_sizes=self.get_next_video_chunk_size()
            video_chunk_remain=len(self.video_trace)-self.req_chunk_id
            around,focuse,state,action,reward,entroy=abr_algorithms.DQN_base(self.last_step_info,next_video_chunk_sizes,video_chunk_remain,self.a3cbatch.get_s_batch(),std_bit)
            self.a3cbatch.add(state,action,reward,entroy)
        elif abr_name=="comyco":
            next_video_chunk_sizes=self.get_next_video_chunk_size()
            video_chunk_remain=len(self.video_trace)-self.req_chunk_id
            around,focuse,state,action,reward,entroy=abr_algorithms.Comyco_base(self.last_step_info,next_video_chunk_sizes,video_chunk_remain,self.a3cbatch.get_s_batch(),std_bit)
            self.a3cbatch.add(state,action,reward,entroy)
        else:
            around,focuse=abr_algorithms.normal()
        # 开始执行ABR
        # 暂时简单的ABR算法，focus tile 选最高质量，around tile 选次高质量；之后再修改
        tile_level = {}
        
        for tile_id in range(100):
            tile_level[tile_id] = self.video_trace[str(self.req_chunk_id)][str(tile_id)][around]

        # for tile_id in around_tile_id:
        #     tile_level[tile_id] = self.video_trace[str(self.req_chunk_id)][str(tile_id)][around]
        tile_level[focus_tile_id] = self.video_trace[str(self.req_chunk_id)][str(focus_tile_id)][focuse]


        self.last_step_info["bit rate"]=around
        print("bit rate:"+around)
        video_chunk_size = sum(tile_level.values())*B_IN_MB # 计算video_chunk_size
        self.last_step_info["video chunk size"]=video_chunk_size
        return around,tile_level, video_chunk_size # 返回每个tile的质量，是一个字典或列表；并计算video_chunk_size


    # 要读实际网络带宽轨迹 这个函数相当于一个step
    def step(self): # 直接给出下一个视频块的大小，具体计算在FoV与ABR结束后结合视野点给出

        broadcasting_delay=0.08
        FoV_normalized_eulerian, FoV_cartesian = self.FoV_predict() # 执行预测视野，
        bit_rate,tile_level, video_chunk_size = self.ABR(FoV_normalized_eulerian)
        throughput=self.net_env.get_throuthput()
        # chunk_delay, broadcasting_delay, end_of_video = self.download(video_chunk_size)
        delay, sleep_time, buffer_size, rebuf, fov_accuracy, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                self.net_env.get_video_chunk(int(bit_rate))
        self.req_chunk_id+=1
        if self.req_chunk_id>=300:
            end_of_video=True
            self.reset()
        self.watching_chunk=self.watching_chunk+ delay/1000- rebuf
        self.watching_chunk_id=int(self.watching_chunk)
        # 更新比特率历史和 rebuffer 历史
        self.bitrate_history.append(int(bit_rate))
        self.rebuffer_history.append(self.rebuffer_time)

        # 计算并记录 QoE 指数
        qoe_index = self.get_qoe_metrics()

        self.real_time=self.base_time+self.net_env.get_last_time()
        self.bandwidth_ptr=self.base_time+self.net_env.get_mahimahi_ptr()

        self.buffer_size=buffer_size
        self.rebuffer_time=rebuf

        # if end_of_video:
        #     self.reset() # step数量在外部控制
        
        return (delay,
                bit_rate,
                qoe_index,
                video_chunk_size,
                broadcasting_delay,
                self.real_time,
                self.req_chunk_id,
                self.watching_chunk_id,
                self.bandwidth_ptr,
                throughput,
                self.buffer_size,
                self.rebuffer_time,
                self.a3cbatch,
                end_of_video)
    
import json
import os
import pandas as pd
import subprocess
def read_FoV_Trace(dataset_name, video, user):
    dataset_path = "../head-motion-prediction/"+dataset_name+"/sampled_dataset/"
    
    path = os.path.join(dataset_path, video, user)
    data = pd.read_csv(path, header=None)
    return data.values[:, 1:]

def show_results(log_file_path,save_fig_path):
    plot_script_path = os.path.join(script_dir, 'plot_results.py')
    # 定义要传递给 plot_results.py 的参数
    plot_time_length = 200

    # 构建命令列表，包含 Python 解释器、脚本路径和参数
    command = [
        sys.executable,
        plot_script_path,
        '--plot_time_length', str(plot_time_length),
        '--file_path', log_file_path,
        '--save_path', save_fig_path
    ]
    if os.path.exists(plot_script_path):
        try:
            # 执行命令
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"运行 plot_results.py 时出现错误: {e}")
    else:
        print(f"未找到 plot_results.py 文件: {plot_script_path}")



if __name__ == "__main__":

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    dataset_name = "Fan_NOSSDAV_17"
    video_name = "coaster"
    user_name = "user21"
    abr="comyco"# bb mpc pensieve dqn comyco
    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(False)

    with open("../Video_Trace/Fan_NOSSDAV_17/coaster.json", 'r', encoding='utf-8') as file:
    # 直接使用json.load函数从文件对象中读取内容并转换为字典
        Video_Trace = json.load(file)  # 读出的key值是str，要转换
    FoV_trace = read_FoV_Trace(dataset_name, video_name, user_name)
    
    
    output_file_path = "../output_dev/"+ dataset_name+"_"+video_name+"_"+user_name+"_"+formatted_time +"_client_test.csv"
    log_file_path="../results/"+ abr+"/"+formatted_time +".txt"
    save_fig_path="../figures/"+ abr+"/"+formatted_time
    headers = ["chunk delay", "vided chunk size","broadcasting delay", "real time", "req chunk id", "watching chunk id", "bandwidth ptr", "buffer size", "rebuffer time", "end of video"]
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
        nn_model = A3C_MODEL
        if abr=="pensieve" and nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")
            # 提前加载fov模型
        data_set = "Fan_NOSSDAV_17"
        FoV_model="pos_only"
        fovmodel = create_pos_only_model(M_WINDOW, H_WINDOW)
        fovmodel.load_weights("../head-motion-prediction/" + data_set + "/" + FoV_model + "/Models_EncDec_eulerian_init_5_in_5_out_13_end_13" + '/weights.hdf5')

        client_env = Environment(all_cooked_time=all_cooked_time,
                        all_cooked_bw=all_cooked_bw,
                        video_trace=Video_Trace,
                        FoV_trace=FoV_trace,
                        M_WINDOW=5,
                        H_WINDOW=13,
                        ABR_algorithm=abr,
                        actor=actor,
                        model=fovmodel,
                        use_true_saliency=False,
                        random_seed=RANDOM_SEED)

        while True:
            step_count += 1 # base on step to decide Graph and Adjacent matrix

            (chunk_delay, 
            bit_rate,
            qoe,
            video_chunk_size,
            broadcasting_delay, 
            real_time, 
            req_chunk_id, 
            watching_chunk_id, 
            bandwidth_ptr,
            bandwidth,
            buffer_size,
            rebuffer_time, 
            a3cbatch ,
            end_of_video) = client_env.step()
            
            # print("chunk delay:", chunk_delay, "real time:", real_time, "req chunk id:", req_chunk_id, "watching chunk id:", watching_chunk_id, "bandwidth ptr:", bandwidth_ptr, "buffer size:", buffer_size, "rebuffer time:", rebuffer_time)
            
            step_info = {
                "delay": chunk_delay,
                "bit rate":bit_rate,
                "qoe":qoe,
                "video chunk size":video_chunk_size,
                "broadcasting delay": broadcasting_delay,
                "real time": real_time,
                "req chunk id": req_chunk_id,
                "watching chunk id": watching_chunk_id,
                "bandwidth ptr": bandwidth_ptr,
                "bandwidth":bandwidth,
                "buffer size": buffer_size,
                "rebuffer time": rebuffer_time,
                "end of video": end_of_video,
                "a3cbatch":a3cbatch
            }
            # print(step_info["video chunk size"])
            client_env.last_step_info=step_info
            # qoe=get_qoe_metrics()
            data = [chunk_delay,video_chunk_size, broadcasting_delay, real_time, req_chunk_id, watching_chunk_id, bandwidth_ptr, buffer_size, rebuffer_time, end_of_video]
            with open(output_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data)
            with open(log_file_path,'a',newline='') as log_file:
                log_file.write((str(real_time) + '\t' +
                           str(bandwidth/1000000) + '\t' +
                           str(qoe) + '\t' +
                           str(rebuffer_time) + '\t' +
                           str(buffer_size) + '\t' +
                           str(VIDEO_BIT_RATE[int(bit_rate)]) + '\n'))
                
            if step_count > 300:
                r_batch=a3cbatch.get_r_batch()
                
                
                break
    #绘图
    show_results(log_file_path,save_fig_path)
