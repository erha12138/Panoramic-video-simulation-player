import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)
# import tensorflow as tf
# import fixed_env as env
import a3c
import dqn
import torch
import comyco
# import load_trace
# import matplotlib.pyplot as plt
import itertools


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 8
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE =[300,720,1080,1260,1440,2160,3120,4080] # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 300
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 5  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log_sim_rlpretrain'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = '../sim/results/nn_model_ep_100.ckpt'
script_dir = os.path.dirname(os.path.abspath(__file__))
A3C_MODEL = os.path.join(script_dir,'./rl_models/pensieve/models_pretrained/pretrain_linear_reward.ckpt')

RESEVOIR = 5  # BB
CUSHION = 10  # BB

TOTAL_VIDEO_CHUNKS = 300
MPC_FUTURE_CHUNK_COUNT = 5

def normal():
    ground=str(5)
    focuse=str(7)
    return ground,focuse

def buffer_base(last_step_info):
    
    ground=str(5)
    focuse=str(6)
    if len(last_step_info)==0:
         return ground,focuse
    buffer_size=last_step_info["buffer size"]
    if buffer_size < RESEVOIR:
            bit_rate = 0
    elif buffer_size >= RESEVOIR + CUSHION:
        bit_rate = A_DIM - 1
    else:
        bit_rate = (A_DIM - 1) * (buffer_size - RESEVOIR) / float(CUSHION)

    bit_rate = int(bit_rate)
    ground=str(bit_rate)

    return ground,focuse

def calculate_state(last_step_info,next_video_chunk_sizes,video_chunk_remain,s_batch,std_bit):
    if len(last_step_info) == 0:
        state = np.zeros((S_INFO, S_LEN))
        return state
    else:
        state = np.array(s_batch[-1], copy=True)

    # dequeue history record
    state = np.roll(state, -1, axis=1)

    
    bit_rate = int(last_step_info["bit rate"])

    delay=last_step_info["delay"]
    buffer_size=last_step_info["buffer size"]
    video_chunk_size=last_step_info["video chunk size"]

    state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
    state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 30 sec
    state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
    state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 30 sec
    state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
    state[5, -1] = std_bit

    return state

def a3c_base(last_step_info,next_video_chunk_sizes,video_chunk_remain,actor,s_batch,std_bit):
    around=str(5)
    focuse=str(6)

    np.random.seed(RANDOM_SEED)

    # assert len(VIDEO_BIT_RATE) == A_DIM

    bit_rate = DEFAULT_QUALITY
    last_bit_rate=last_step_info["bit rate"]
    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1
    reward=0.0
    entropy=0.0
    state=calculate_state(last_step_info,next_video_chunk_sizes,video_chunk_remain,s_batch,std_bit)

    rebuf=last_step_info["rebuffer time"]

    action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
    action_cumsum = np.cumsum(action_prob)
    bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()


    # reward is video quality - rebuffer penalty - smoothness
    reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                        VIDEO_BIT_RATE[int(last_bit_rate)]) / M_IN_K

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1
    entropy=a3c.compute_entropy(action_prob[0])
    around=str(bit_rate)
    return around,focuse,state,action_vec,reward,entropy


def mpc_base(last_step_info,next_video_chunk_sizes,video_chunk_remain,s_batch,std_bit):
    around=str(5)
    focuse=str(6)
    np.random.seed(RANDOM_SEED)
    CHUNK_COMBO_OPTIONS = []
    past_errors = []
    past_bandwidth_ests = []

    last_bit_rate = last_step_info["bit rate"]
    

    # make chunk combination options
    for combo in itertools.product([0,1,2,3,4,5,6,7], repeat=5):
        CHUNK_COMBO_OPTIONS.append(combo)

    state=calculate_state(last_step_info,next_video_chunk_sizes,video_chunk_remain,s_batch,std_bit)

    buffer_size=last_step_info["buffer size"]
    rebuf=last_step_info["rebuffer time"]
    # reward is video quality - rebuffer penalty
    
    # ================== MPC =========================
    curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
    if ( len(past_bandwidth_ests) > 0 ):
        curr_error  = abs(past_bandwidth_ests[-1]-state[3,-1])/float(state[3,-1])
    past_errors.append(curr_error)

    # pick bitrate according to MPC           
    # first get harmonic mean of last 5 bandwidths
    past_bandwidths = state[2,-5:]
    while past_bandwidths[0] == 0.0:
        past_bandwidths = past_bandwidths[1:]
    #if ( len(state) < 5 ):
    #    past_bandwidths = state[3,-len(state):]
    #else:
    #    past_bandwidths = state[3,-5:]
    bandwidth_sum = 0
    for past_val in past_bandwidths:
        bandwidth_sum += (1/float(past_val))
    harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

    # future bandwidth prediction
    # divide by 1 + max of last 5 (or up to 5) errors
    max_error = 0
    error_pos = -5
    if ( len(past_errors) < 5 ):
        error_pos = -len(past_errors)
    max_error = float(max(past_errors[error_pos:]))
    future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
    past_bandwidth_ests.append(harmonic_bandwidth)


    # future chunks length (try 4 if that many remaining)
    last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain)
    future_chunk_length = MPC_FUTURE_CHUNK_COUNT
    if ( TOTAL_VIDEO_CHUNKS - last_index < 5 ):
        future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index

    # all possible combinations of 5 chunk bitrates (9^5 options)
    # iterate over list and for each, compute reward and store max reward combination
    max_reward = -100000000
    best_combo = ()
    start_buffer = buffer_size
    #start = time.time()
    for full_combo in CHUNK_COMBO_OPTIONS:
        combo = full_combo[0:future_chunk_length]
        # calculate total rebuffer time for this combination (start with start_buffer and subtract
        # each download time and add 2 seconds in that order)
        curr_rebuffer_time = 0
        curr_buffer = start_buffer
        bitrate_sum = 0
        smoothness_diffs = 0
        last_quality = int( last_step_info["bit rate"] )
        for position in range(0, len(combo)):
            chunk_quality = combo[position]
            index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
            download_time = (next_video_chunk_sizes[chunk_quality]/1000000.)/future_bandwidth # this is MB/MB/s --> seconds
            if ( curr_buffer < download_time ):
                curr_rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0
            else:
                curr_buffer -= download_time
            curr_buffer += 4
            bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
            smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
            # bitrate_sum += BITRATE_REWARD[chunk_quality]
            # smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])
            last_quality = chunk_quality
        # compute reward for this combination (one reward per 5-chunk combo)
        # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s
        
        reward = (bitrate_sum/1000.) - (REBUF_PENALTY*curr_rebuffer_time) - (smoothness_diffs/1000.)
        # reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)


        if ( reward >= max_reward ):
            if (best_combo != ()) and best_combo[0] < combo[0]:
                best_combo = combo
            else:
                best_combo = combo
            max_reward = reward
            # send data to html side (first chunk of best combo)
            send_data = 0 # no combo had reward better than -1000000 (ERROR) so send 0
            if ( best_combo != () ): # some combo was good
                send_data = best_combo[0]

    bit_rate = send_data
    around=str(bit_rate)
    reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                        VIDEO_BIT_RATE[int(last_bit_rate)]) / M_IN_K
    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    return around, focuse, state, action_vec
# def get_chunk_size():
#     return 0

def DQN_base(last_step_info,next_video_chunk_sizes,video_chunk_remain,s_batch,std_bit):
    model_path="E:/abr-fov-dev/FoV-ABR-master/Client_logic/rl_models/DQN/model/DQN_model_150000.pth"
    around=str(5)
    focuse=str(6)
    input_dim = [S_INFO, S_LEN]
    output_dim = A_DIM
    agent = dqn.DQNAgent(input_dim, output_dim)
    agent.eval_model.load_state_dict(torch.load(model_path))
    agent.eval_model.eval()  # 设置模型为评估模式
    bit_rate = DEFAULT_QUALITY
    last_bit_rate=last_step_info["bit rate"]
    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1
    reward=0.0
    entropy=0.0
    state=calculate_state(last_step_info,next_video_chunk_sizes,video_chunk_remain,s_batch,std_bit)

    rebuf=last_step_info["rebuffer time"]

    bit_rate = agent.act(np.reshape(state, (1, S_INFO, S_LEN)))


    # reward is video quality - rebuffer penalty - smoothness
    reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                        VIDEO_BIT_RATE[int(last_bit_rate)]) / M_IN_K

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1
    around=str(bit_rate)
    return around,focuse,state,action_vec,reward,entropy
def Comyco_base(last_step_info,next_video_chunk_sizes,video_chunk_remain,s_batch,std_bit):
    model_path="E:/abr-fov-dev/FoV-ABR-master/Client_logic/rl_models/Comyco/model/Comyco_model_ep_300.pth"
    around=str(5)
    focuse=str(6)
    agent = comyco.libcomyco(S_INFO, S_LEN, A_DIM)
    
    model_params = torch.load(model_path)
    agent.net.load_state_dict(model_params)
    agent.net.eval()  # 设置模型为评估模式
    # print("模型加载成功，开始测试..."+model_path)
    bit_rate = DEFAULT_QUALITY
    last_bit_rate=last_step_info["bit rate"]
    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1
    reward=0.0
    entropy=0.0
    state=calculate_state(last_step_info,next_video_chunk_sizes,video_chunk_remain,s_batch,std_bit)

    rebuf=last_step_info["rebuffer time"]

    _ , bit_rate = agent.predict(np.reshape(state, (1, S_INFO, S_LEN)))


    # reward is video quality - rebuffer penalty - smoothness
    reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                        VIDEO_BIT_RATE[int(last_bit_rate)]) / M_IN_K

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1
    around=str(bit_rate)
    return around,focuse,state,action_vec,reward,entropy