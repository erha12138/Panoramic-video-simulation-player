from dataclasses import dataclass
import random

class AerialType:
    UAV = 0
    BALLOON = 1
    HAPS = 2


@dataclass
class Vector:
    x: float
    y: float
    z: float


@dataclass
class Vector3D:
    x: float
    y: float
    z: float



class AerialNode:
    def __init__(self, position : Vector3D, aerial_type = AerialType.UAV, bandwidth = 2, id = 0):
        self.node_type = "Aerial"
        self.id = id
        self.m_position = position
        self.m_type = aerial_type
        self.bandwidth =  bandwidth # 可以随机设置的，一路上链路的带宽最低值就是实际带宽
        if self.m_type == 0:
            self.m_velocity = Vector(random.random(0.03,0.08), random.random(0.03,0.08), 0.0) # 乱飞即可，初始化一个
            self.visibility = 0.5 # km
        elif self.m_type == 1:
            self.m_velocity = Vector(random.random(10,20), random.random(10,20), 0.0) # 乱飞即可，初始化一个
            self.visibility = 500 # km
        else:
            self.m_velocity = Vector(random.random(5,10), random.random(5,10), 0.0) # 乱飞即可，初始化一个
            self.visibility = 500


        

    def SetPosition(self, position):
        self.m_position = position

    def GetPosition(self):
        return self.m_position

    def SetVelocity(self, velocity):
        self.m_velocity = velocity

    def GetVelocity(self):
        return self.m_velocity