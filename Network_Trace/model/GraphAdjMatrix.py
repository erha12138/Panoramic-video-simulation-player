from .ground_node import GroundNode, Vector3D
from .air_node import AerialNode
from .space_node import SpaceNode
import random
import sys
# propagation delay parameters
AIR_GROUND_PARA = 1 # 0.2km
AIR_SPACE_PARA = 0.001 # 1000km
GROUND_SPACE_PARA = 0.001
SPACE_SPACE = 0.05 # 200km
CLIENT_X = 0
CLIENT_Y = 0
SERVER_X = 2000
SERVER_Y = 2000

# get all position of all nodes at any stamp
def mobility_control(node, stamp):
    node.m_position.x += stamp * node.m_velocity.x
    node.m_position.y += stamp * node.m_velocity.y
    node.m_position.z += stamp * node.m_velocity.z
    if node.m_position.x >= SERVER_X:
        node.m_position.x = 0
    if node.m_position.y >= SERVER_Y:
        node.m_position.y = 0

# Assume client (0,0,0) to server (1000,1000,0)
# 形成每一时刻的节点图
# 1、generate all node and initialize
# 2、Traverse all node and generate adjacency matrix

# 不会写其实是对于路由算法不了解！！先看看路由算法

# 主要生成邻接矩阵，每个时刻都有自己的邻接矩阵
# 下一步写邻接矩阵逻辑，将每个节点一个一个遍历处理，要生成两个邻接矩阵
class GraphAdjMatrix:
    def __init__(self, vertices, ground_node_num, air_node_num, space_node_num):
        self.vertices = vertices
        self.num_vertices = len(vertices)
        self.bandwidth_matrix = [[0] * self.num_vertices for _ in range(self.num_vertices)]
        self.propagation_matrix = [[float('inf')] * self.num_vertices for _ in range(self.num_vertices)]
        for i in range(self.num_vertices):
            self.bandwidth_matrix[i][i] = 0
            self.propagation_matrix[i][i] = [float('inf')]

        self.ground_node_num = ground_node_num
        self.air_node_num = air_node_num
        self.space_node_num = space_node_num

        self.propagation_para_map = self.get_propagation_map()
        self.make_edge()
    
    def get_propagation_map(self):
        temp_weight = 2
        # could be modify
        Ground_Ground = 0.05 * temp_weight
        Ground_Aerial = 0.2 * temp_weight
        Ground_Space = 0.01 * temp_weight
        Aerial_Aerial = 0.2 * temp_weight
        Aerial_Space = 0.02 * temp_weight
        Space_Space = 0.1 * temp_weight

        type_propagation_map = {}
        type_propagation_map["Ground"] = {}
        type_propagation_map["Ground"]["Ground"] = Ground_Ground # parameter
        type_propagation_map["Ground"]["Aerial"] = Ground_Aerial
        type_propagation_map["Ground"]["Space"] =  Ground_Space
        type_propagation_map["Aerial"] = {}

        type_propagation_map["Aerial"]["Ground"] = Ground_Aerial
        type_propagation_map["Aerial"]["Aerial"] = Aerial_Aerial
        type_propagation_map["Aerial"]["Space"] = Aerial_Space
        type_propagation_map["Space"] = {}
        type_propagation_map["Space"]["Ground"] = Ground_Space
        type_propagation_map["Space"]["Aerial"] = Aerial_Space
        type_propagation_map["Space"]["Space"] = Space_Space
        
        return type_propagation_map
    
    def get_distance(self, position1:Vector3D, position2:Vector3D):
        return ((position1.x - position2.x)**2 + (position1.y - position2.y)**2 + (position1.z - position2.z)**2) ** 0.5
    
    def get_propagation_delay_and_bandwidth(self, u, v):
        if u == v:
            return 0, float('inf')
        
        node1 = self.vertices[u]
        node2 = self.vertices[v]
        node_type_1 = node1.node_type
        node_type_2 = node2.node_type
        if node_type_1 == "End":
            node_type_1 = "Ground"
        if node_type_2 == "End":
            node_type_2 = "Ground"
        
        distance = self.get_distance(node1.m_position,node2.m_position)
        propagation_delay = self.propagation_para_map[node_type_1][node_type_2] * distance
        # wireless link consider visibility

        if node_type_1 != "Ground" or node_type_2 != "Ground":
            visibility = max(node1.visibility, node2.visibility)
            if distance > visibility:
                return 0, float('inf')
        bandwidth = min(node1.bandwidth,node2.bandwidth)
        return bandwidth, propagation_delay

    def make_edge(self): # 
        # handle client: first hop and the ground especially
        for u in range(self.ground_node_num + 1):
            if u < 4:
                for v in range(0,4):
                    bandwidth, propagation_delay = self.get_propagation_delay_and_bandwidth(u, v)
                    self.bandwidth_matrix[u][v] = self.bandwidth_matrix[v][u] = bandwidth
                    self.propagation_matrix[u][v] = self.propagation_matrix[v][u] = propagation_delay
            if u >= 4 and u < 8:
                for v in range(1,11):
                    bandwidth, propagation_delay = self.get_propagation_delay_and_bandwidth(u, v)
                    self.bandwidth_matrix[u][v] = self.bandwidth_matrix[v][u] = bandwidth
                    self.propagation_matrix[u][v] = self.propagation_matrix[v][u] = propagation_delay
            if u >= 8 and u < 11:
                for v in range(4,14):
                    bandwidth, propagation_delay = self.get_propagation_delay_and_bandwidth(u, v)
                    self.bandwidth_matrix[u][v] = self.bandwidth_matrix[v][u] = bandwidth
                    self.propagation_matrix[u][v] = self.propagation_matrix[v][u] = propagation_delay
            if u >= 11 and u < 14:
                for v in [i for i in range(8,14)] + [60]:
                    bandwidth, propagation_delay = self.get_propagation_delay_and_bandwidth(u, v)
                    self.bandwidth_matrix[u][v] = self.bandwidth_matrix[v][u] = bandwidth
                    self.propagation_matrix[u][v] = self.propagation_matrix[v][u] = propagation_delay

        for u in range(self.ground_node_num + 1, self.num_vertices):
            for v in range(self.ground_node_num + 1, self.num_vertices):
                bandwidth, propagation_delay = self.get_propagation_delay_and_bandwidth(u, v)
                self.bandwidth_matrix[u][v] = self.bandwidth_matrix[v][u] = bandwidth
                self.propagation_matrix[u][v] = self.propagation_matrix[v][u] = propagation_delay

        for u in range(self.ground_node_num + 1):
            for v in range(self.ground_node_num + 1, self.num_vertices - 1):
                bandwidth, propagation_delay = self.get_propagation_delay_and_bandwidth(u, v)
                self.bandwidth_matrix[u][v] = self.bandwidth_matrix[v][u] = bandwidth
                self.propagation_matrix[u][v] = self.propagation_matrix[v][u] = propagation_delay

        print(self.bandwidth_matrix)
        print(self.propagation_matrix)

    def get_new_AdjMatrix(self, vertices):
        self.vertices = vertices
        self.make_edge()


# id = 0 client

class End_Device():
    def __init__(self, position:Vector3D, id:int):
        self.m_position = position
        self.id = 0
        self.node_type = "End"
        self.bandwidth = random.uniform(5,12)
        self.visibility = 0
        self.m_velocity = Vector3D(0.0, 0.0, 0.0)

# 四颗卫星，卫星直连地面基站更慢一点，经过无人机中继会更快一些
def initialize_graph(): 
    # Ground node 
    # because of the wire link, no need for visibility, just design the adjacent relationship        
    groundNode_list = [] # first hop
    for i in range(1,4): # fist hop number
        groundNode_list.append(GroundNode(position=Vector3D(random.uniform(CLIENT_X,SERVER_X/5),random.uniform(CLIENT_Y,SERVER_Y/5),0), bandwidth=random.uniform(3,10),id=i))
    for i in range(4,8):
        groundNode_list.append(GroundNode(position=Vector3D(random.uniform(CLIENT_X+SERVER_X/5,SERVER_X/2),random.uniform(CLIENT_Y+SERVER_Y/5,SERVER_Y/2),0), bandwidth=random.uniform(3,10),id=i))
    for i in range(8,11):
        groundNode_list.append(GroundNode(position=Vector3D(random.uniform(SERVER_X/2,SERVER_X/2+SERVER_X/5),random.uniform(SERVER_Y/2,SERVER_Y/2+SERVER_Y/5),0), bandwidth=random.uniform(3,10),id=i))
    for i in range(11,14):
        groundNode_list.append(GroundNode(position=Vector3D(random.uniform(SERVER_X/2+SERVER_X/5,SERVER_X),random.uniform(SERVER_Y/2+SERVER_Y/5,SERVER_Y),0), bandwidth=random.uniform(3,10),id=i))
    ground_node_num = len(groundNode_list)
    # Air node
    aerialNode_list = []
    for i in range(14,50): # UAV
        aerialNode_list.append(AerialNode(position=Vector3D(random.uniform(CLIENT_X,SERVER_X),random.uniform(CLIENT_Y,SERVER_Y),random.uniform(0.1,0.2)),aerial_type=0,bandwidth=random.uniform(2,8),id=i))
    for i in range(50,53): # balloon
        aerialNode_list.append(AerialNode(position=Vector3D(random.uniform(CLIENT_X,SERVER_X),random.uniform(CLIENT_Y,SERVER_Y),random.uniform(17,22)),aerial_type=1,bandwidth=random.uniform(3,9),id=i))
    for i in range(53,56): # HAPS
        aerialNode_list.append(AerialNode(position=Vector3D(random.uniform(CLIENT_X,SERVER_X),random.uniform(CLIENT_Y,SERVER_Y),random.uniform(18,50)),aerial_type=2,bandwidth=random.uniform(3,9),id=i))
    air_node_num = len(aerialNode_list)

    # Space Node
    spaceNode_list = []
    for i in range(56,60): # LEO 
        spaceNode_list.append(SpaceNode(position=Vector3D(random.uniform(CLIENT_X,SERVER_X),random.uniform(CLIENT_Y,SERVER_Y),random.uniform(600,800)),bandwidth=random.uniform(3,10),id=i))
    space_node_num = len(spaceNode_list)

    all_node = groundNode_list + aerialNode_list + spaceNode_list

    client = End_Device(position=Vector3D(CLIENT_X,CLIENT_X,0),id=0)
    server = End_Device(position=Vector3D(SERVER_X,SERVER_Y,0),id=60)
    all_node.insert(0, client)
    all_node.append(server)

    return all_node, ground_node_num, air_node_num, space_node_num 

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
    node_list, ground_node_num, air_node_num, space_node_num = initialize_graph()
    # print(node_list, len(node_list))
    adjMatrix = GraphAdjMatrix(node_list, ground_node_num, air_node_num, space_node_num)
    start = 0
    end = len(adjMatrix.propagation_matrix) - 1
    path, a = dijkstra(adjMatrix.propagation_matrix, start, end)
    
    real_bandwidth = float('inf')
    for i in range(len(path)-1):
        part_bandwidth = adjMatrix.bandwidth_matrix[i][i+1]
        real_bandwidth = real_bandwidth if real_bandwidth <= part_bandwidth else part_bandwidth
        
    print(path, a, real_bandwidth)
