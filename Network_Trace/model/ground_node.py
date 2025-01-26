from typing import Optional
from scapy.all import Packet


class Vector3D:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z # 一直为0

class GroundNode:
    def __init__(self, position : Vector3D, bandwidth = 10, id = 0):
        self.node_type = "Ground" # 
        self.id = id
        self.m_position = position
        self.bandwidth = bandwidth


    def SetPosition(self, position: Vector3D):
        self.m_position = position

    def GetPosition(self) -> Vector3D:
        return self.m_position
