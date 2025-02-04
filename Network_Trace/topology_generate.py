import sys
import os
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
os.chdir(current_dir) # 切换到当前文件所在目录
sys.path.append('../')
from model.GraphAdjMatrix import initialize_graph, GraphAdjMatrix

def dijkstra(adj_matrix, start, end):
    num_nodes = len(adj_matrix)
    # 初始化距离数组，将起始节点到自身的距离设为 0，其他节点设为无穷大
    dist = [sys.maxsize] * num_nodes
    dist[start] = 0
    # 初始化前驱节点数组，用于记录最短路径
    prev = [-1] * num_nodes
    # 记录节点是否已被访问
    visited = [False] * num_nodes

    for _ in range(num_nodes):
        # 从未访问的节点中选择距离最小的节点
        min_dist = sys.maxsize
        min_index = -1
        for v in range(num_nodes):
            if not visited[v] and dist[v] < min_dist:
                min_dist = dist[v]
                min_index = v

        # 如果没有找到未访问的节点，退出循环
        if min_index == -1:
            break

        # 标记当前节点为已访问
        visited[min_index] = True

        # 更新与当前节点相邻节点的距离
        for v in range(num_nodes):
            if not visited[v] and adj_matrix[min_index][v] > 0 and dist[min_index] + adj_matrix[min_index][v] < dist[v]:
                dist[v] = dist[min_index] + adj_matrix[min_index][v]
                prev[v] = min_index

    # 回溯最短路径
    path = []
    at = end
    while at != -1:
        path.append(at)
        at = prev[at]
    path.reverse()

    if path[0] != start:
        print("未找到从节点 {} 到节点 {} 的路径".format(start, end))
        return None, None

    return path, dist[end]

if __name__ == "__main__":
    node_list,ground_node_num, air_node_num, space_node_num = initialize_graph()
    # print(node_list, len(node_list))
    adjMatrix = GraphAdjMatrix(node_list,ground_node_num, air_node_num, space_node_num)
    start = 0
    end = len(adjMatrix.propagation_matrix)
    path, a = dijkstra(adjMatrix.propagation_matrix, start, end)
    print(path, a)
