from ground_node import GroundNode, Vector3D
from air_node import AerialNode
from space_node import SpaceNode
import random
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

# Assume client (0,0,0) to server (1000,1000,0)
# 形成每一时刻的节点图
# 1、generate all node and initialize
# 2、Traverse all node and generate adjacency matrix

# 不会写其实是对于路由算法不了解！！先看看路由算法

# 主要生成邻接矩阵，每个时刻都有自己的邻接矩阵
# 下一步写邻接矩阵逻辑，将每个节点一个一个遍历处理
class GraphAdjMatrix:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.matrix = [[float('inf')] * num_vertices for _ in range(num_vertices)]
        for i in range(num_vertices):
            self.matrix[i][i] = 0

    def add_edge(self, u, v, weight):
        self.matrix[u][v] = weight

    def get_adj_matrix(self):
        return self.matrix

# id = 0 client

# 四颗卫星，卫星直连地面基站更慢一点，经过无人机中继会更快一些
def initialize_graph(): 
    # Ground node 
    # because of the wire link, no need for visibility, just design the adjacent relationship        
    groundNode_list = [] # first hop
    for i in range(1,4): # fist hop number
        groundNode_list.append(GroundNode(position=Vector3D(random.random(CLIENT_X,SERVER_X/5),random.random(CLIENT_Y,SERVER_Y/5),0), bandwidth=random.random(3,10),id=i))
    for i in range(4,8):
        groundNode_list.append(GroundNode(position=Vector3D(random.random(CLIENT_X+SERVER_X/5,SERVER_X/2),random.random(CLIENT_Y+SERVER_Y/5,SERVER_Y/2),0), bandwidth=random.random(3,10),id=i))
    for i in range(8,11):
        groundNode_list.append(GroundNode(position=Vector3D(random.random(SERVER_X/2,SERVER_X/2+SERVER_X/5),random.random(SERVER_Y/2,SERVER_Y/2+SERVER_Y/5),0), bandwidth=random.random(3,10),id=i))
    for i in range(11,14):
        groundNode_list.append(GroundNode(position=Vector3D(random.random(SERVER_X/2+SERVER_X/5,SERVER_X),random.random(SERVER_Y/2+SERVER_Y/5,SERVER_Y),0), bandwidth=random.random(3,10),id=i))

    # Air node
    aerialNode_list = []
    for i in range(14,50): # UAV
        aerialNode_list.append(AerialNode(position=Vector3D(random.random(CLIENT_X,SERVER_X),random.random(CLIENT_Y,SERVER_Y),random.random(0.1,0.2)),aerial_type=0,bandwidth=random.random(2,8),id=i))
    for i in range(50,53): # balloon
        aerialNode_list.append(AerialNode(position=Vector3D(random.random(CLIENT_X,SERVER_X),random.random(CLIENT_Y,SERVER_Y),random.random(17,22)),aerial_type=1,bandwidth=random.random(3,9),id=i))
    for i in range(53,56): # HAPS
        aerialNode_list.append(AerialNode(position=Vector3D(random.random(CLIENT_X,SERVER_X),random.random(CLIENT_Y,SERVER_Y),random.random(18,50)),aerial_type=2,bandwidth=random.random(3,9),id=i))

    # Space Node
    spaceNode_list = []
    for i in range(56,60): # LEO 
        spaceNode_list.append(SpaceNode(position=Vector3D(random.random(CLIENT_X,SERVER_X),random.random(CLIENT_Y,SERVER_Y),random.random(600,800)),bandwidth=random.random(3,10),id=i))
    
    all_node = groundNode_list + aerialNode_list + spaceNode_list
    
    return all_node


