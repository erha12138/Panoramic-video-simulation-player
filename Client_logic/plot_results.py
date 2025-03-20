import pandas as pd
import matplotlib.pyplot as plt
import argparse

# 创建参数解析器
parser = argparse.ArgumentParser(description='Plot results from a TSV file.')
# 添加参数
parser.add_argument('--plot_time_length', type=int, default=200, help='Number of data points to plot.')
parser.add_argument('--file_path', type=str, default='E:/abr-fov-dev/FoV-ABR-master/Client_logic/rl_models/DQN/results_test/test_results.txt', help='Path to the TSV file.')
parser.add_argument('--save_path', type=str, default='E:/abr-fov-dev/FoV-ABR-master/Client_logic/rl_models/DQN/results_test/plot_results.png', help='Path to save the plot.')

# 解析参数
args = parser.parse_args()


save_path = args.save_path
file_path=args.file_path

try:

    plot_time_length=200
    # 读取 txt 文件，这里 txt 文件实际是 TSV 格式，使用制表符分隔
    
    data = pd.read_csv(file_path, sep='\t', header=None)
    # 提取带宽（第一列）和比特率（第二列），并只选取前 500 条数据
    bandwidth = data[1][:plot_time_length]
    qoe=data[2][:plot_time_length]
    rebuf=data[3][:plot_time_length]
    buffer_size=data[4][:plot_time_length]
    bitrate = data[5][:plot_time_length]
    
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 100

    # 创建一个包含 4 行 1 列的子图布局
    fig, axes = plt.subplots(4, 1, figsize=(18.42, 10.21))

    # 第一个子图：绘制带宽和比特率折线图
    ax1 = axes[0]
    ax2 = ax1.twinx()
    line1 = ax1.plot(bandwidth, label='Bandwidth (MB/S)', color='blue')
    ax1.set_ylabel('Bandwidth (MB/S)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    line2 = ax2.plot(bitrate, label='Bitrate (Kbps)', color='red')
    ax2.set_ylabel('Bitrate (Kbps)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.set_title('Bandwidth and Bitrate Over Time on DQN')

    # 第二个子图：绘制奖励折线图
    ax3 = axes[1]
    ax3.plot(qoe, label='QoE', color='green')
    ax3.set_ylabel('QoE')
    ax3.legend(loc='upper left')
    ax3.set_title('QoE Over Time')

    # 第三个子图：绘制卡顿折线图
    ax4 = axes[2]
    ax4.plot(rebuf, label='Rebuffering', color='orange')
    ax4.set_ylabel('Rebuffering')
    ax4.legend(loc='upper left')
    ax4.set_title('Rebuffering Over Time on DQN')

    # 第四个子图：绘制缓冲区大小折线图
    ax5 = axes[3]
    ax5.plot(buffer_size, label='Buffer Size', color='purple')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Buffer Size')
    ax5.legend(loc='upper left')
    ax5.set_title('Buffer Size Over Time on DQN')

    # 自动调整子图布局
    plt.tight_layout()

    # 显示图形
    plt.savefig(save_path)
    print("模型已保存。。。"+save_path)
    # plt.show()
    
except FileNotFoundError:
    print(f"未找到文件: {file_path}，请检查文件路径是否正确。")
except Exception as e:
    print(f"读取文件或绘图过程中出现错误: {e}")