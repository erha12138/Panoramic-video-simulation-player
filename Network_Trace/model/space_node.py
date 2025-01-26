# space_node.py
import math
import random
class Vector3D:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 800.0): # 高度默认800m
        self.x = x
        self.y = y
        self.z = z

class Vector:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y

class ConstantVelocityMobilityModel(): # 速度方向
    def __init__(self):
        self.velocity = Vector(0.0, 0.0)

    def SetVelocity(self, velocity):
        self.velocity = velocity

# 设置带宽，与传播时延，传播时延受地理位置影响；
class SpaceNode:
    def __init__(self, position : Vector3D,bandwidth = 5, id = 0):
        """
        构造函数，用于初始化 SpaceNode 对象。
        :param radius: 空间节点的半径，默认为 1.0
        """
        self.node_type = "Space" # 
        self.id = id
        self.m_position = position
        self.m_velocity = Vector3D(random.random(20,60), random.random(20,60), 0.0) # 
        self.bandwidth = bandwidth
        self.visibility = 1000 # km

    def SetPosition(self, position):
        self.m_position = position

    def GetPosition(self):
        return self.m_position

    def SetVelocity(self, velocity):
        self.m_velocity = velocity

    def GetVelocity(self):
        return self.m_velocity
