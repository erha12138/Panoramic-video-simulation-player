import sys
import os
import pandas as pd
import random
import json

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
os.chdir(current_dir) # 切换到当前文件所在目录

V360P_30_PER_CHUNK_1S = 4 # Mbps
V720P_60_PER_CHUNK_1S = 6 # Mbps
V1080P_30_PER_CHUNK_1S = 10 # Mbps
V1080P_60_PER_CHUNK_1S = 12 # Mbps
V1440P_30_PER_CHUNK_1S = 15 # Mbps
V1440P_60_PER_CHUNK_1S = 24 # Mbps
V2160P_30_PER_CHUNK_1S = 30 # Mbps
V2160P_60_PER_CHUNK_1S = 35 # Mbps
TILE_NUM = 10 * 10

DATASET_NAME = "Fan_NOSSDAV_17"
VIDEO_INFO_DIC = "../head-motion-prediction/" + DATASET_NAME + "/sampled_dataset/"
VIDEO_NAME = "coaster"

# VIDEO_INFO_DIC = "../head-motion-prediction/Xu_PAMI_18/sampled_dataset/"
# VIDEO_NAME = "CandyCarnival"

VIDEO_INFO_PATH = VIDEO_INFO_DIC + VIDEO_NAME

# Returns the ids of the videos in the dataset
def get_video_ids(sampled_dataset_folder):
    list_of_videos = [o for o in os.listdir(sampled_dataset_folder) if not o.endswith('.gitkeep')]
    # Sort to avoid randomness of keys(), to guarantee reproducibility
    list_of_videos.sort()
    return list_of_videos


def get_users_per_video(sampled_dataset_folder):
    videos = get_video_ids(sampled_dataset_folder)
    users_per_video = {}
    for video in videos:
        users_per_video[video] = [user for user in os.listdir(os.path.join(sampled_dataset_folder, video))]
    return users_per_video

def read_sampled_positions_for_trace(sampled_dataset_folder, video, user):
    path = os.path.join(sampled_dataset_folder, video, user)
    data = pd.read_csv(path, header=None)
    return data.values[:, 1:]

# 读这个视频的所有用户观看长度，取最长值，视频长度信息就以这个值为准
def get_video_duration(sampled_dataset_folder, video):
    users = get_users_per_video(sampled_dataset_folder)[video]
    max_duration = 0
    for user in users:
        data = read_sampled_positions_for_trace(sampled_dataset_folder, video, user)
        max_duration = max(max_duration, len(data))
    return max_duration

def get_video_noise(quality_level):
    return quality_level / random.uniform(2, 10)


# 在video_name的文件夹下生成以video_duration长度的视频数据集，依照video_chunk_size分别为不同质量等级的chunk生成size，每个chunk被切分为10*10的tile
def generate_video_dataset(video_name, video_duration): # 从FoV数据库中读入
    chunk_size = [V360P_30_PER_CHUNK_1S, V720P_60_PER_CHUNK_1S, V1080P_30_PER_CHUNK_1S, V1080P_60_PER_CHUNK_1S, V1440P_30_PER_CHUNK_1S,V1440P_60_PER_CHUNK_1S, V2160P_30_PER_CHUNK_1S, V2160P_60_PER_CHUNK_1S]
    output_dict = os.path.join("./", DATASET_NAME)
    output_file = os.path.join(output_dict, video_name + ".json")
    if not os.path.exists(output_dict):
        os.makedirs(output_dict)  # 存在output_dict里，以不同的video为文件名
    video_trace = {} # 每个 chunk 一个 
    for chunk_id in range(video_duration): # 用json格式存储
        video_trace[chunk_id] = {}
        for tile_id in range(TILE_NUM):
            video_trace[chunk_id][tile_id] = {}
            for quality_level in range(len(chunk_size)):
                video_quality = chunk_size[quality_level]
                video_trace[chunk_id][tile_id][quality_level] = (video_quality + get_video_noise(video_quality)) / TILE_NUM
    # video_trace 存储为json格式
    with open(output_file, "w") as f:
        json.dump(video_trace, open(output_file, "w"))

if __name__ == "__main__":
    video_duration = get_video_duration(VIDEO_INFO_DIC, VIDEO_NAME)
    video_trace = generate_video_dataset(VIDEO_NAME, video_duration)
    # print(video_duration)