import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)
import tensorflow as tf
import fixed_env as env
import a3c
import load_trace
import matplotlib.pyplot as plt


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
RESEVOIR = 5  # BB
CUSHION = 10  # BB
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
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
NN_MODEL = os.path.join(script_dir,'./models/pretrain_linear_reward.ckpt')


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

def a3c_base(last_step_info,next_video_chunk_sizes,video_chunk_remain,actor,s_batch):
    ground=str(5)
    focuse=str(6)

    np.random.seed(RANDOM_SEED)

    # assert len(VIDEO_BIT_RATE) == A_DIM

    bit_rate = DEFAULT_QUALITY
    last_bit_rate=bit_rate
    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1
    reward=0.0
    entropy=0.0
    # retrieve previous state
    if len(last_step_info) == 0:
        state = np.zeros((S_INFO, S_LEN))
        return ground,focuse,state,action_vec,reward,entropy
    else:
        state = np.array(s_batch[-1], copy=True)

    # dequeue history record
    state = np.roll(state, -1, axis=1)

    delay=last_step_info["chunk delay"]
    buffer_size=last_step_info["buffer size"]
    rebuf=last_step_info["rebuffer time"]
    video_chunk_size=last_step_info["req chunk id"]
    # next_video_chunk_sizes=next_video_chunk_sizes
    end_of_video=last_step_info["end of video"]
    # video_chunk_remain=

    # reward is video quality - rebuffer penalty - smoothness
    reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                        VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

    
    # this should be S_INFO number of terms
    state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
    state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
    state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
    state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
    state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
    state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

    action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
    action_cumsum = np.cumsum(action_prob)
    bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
    # Note: we need to discretize the probability into 1/RAND_RANGE steps,
    # because there is an intrinsic discrepancy in passing single state and batch states
    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1
    entropy=a3c.compute_entropy(action_prob[0])
    ground=str(bit_rate)
    return ground,focuse,state,action_vec,reward,entropy
