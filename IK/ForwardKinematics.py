"""
author: Yanyan Yuan
date: 2023.11.23
discription: this scripts is to solve FK of Kirin arm(7of)
input params: 
        theta_t: target joints angles(7x1)

date: 2024.12.12
discription: this scripts is to solve FK of Kirin arm(4of)
input params: 
        theta_t: target joints angles(4x1)
"""
import numpy as np
from numpy import sin, cos
from scipy.spatial.transform import Rotation as R

class ForwardKinematics():
    def __init__(self, theta_t,
                 a = [0.0, 0.0, 0.0, 0.0, 0.292, 0.012, 0.0],
                 d = [0.159, -0.1255, -0.315, 0.0, 0.0, 0.0, -0.0692]):
        # # 改进型DH参数表
        # self.alpha = [np.pi/2, np.pi/2, -np.pi/2, np.pi/2, 0.0, np.pi/2, np.pi/2]
        # self.a = [0.0, 0.0, 0.0, 0.0, 0.292, 0.012, 0.0]
        # self.d = [0.159, 0.1255, 0.315, 0.0, 0.0, 0.0, 0.0692]
        # # self.theta = np.array([np.pi/2, np.pi/2, 0.0, np.pi/2, 0.0, np.pi/2, 0.0])
        # self.theta = np.array([np.pi/2, np.pi/2, 0.0, 0.0, 0.0, np.pi/2, 0.0])
        
        # for i in range(len(self.alpha)):
        #     if i  == 0 or i == 5:
        #        self.theta[i] = self.theta[i] + theta_t[i]
        #     else:
        #        self.theta[i] = self.theta[i] - theta_t[i]

        # 基于机械臂实际转动正方形重新定义的改进型DH参数表
        # 七自由度
        # self.alpha = [-np.pi/2, np.pi/2, -np.pi/2, np.pi/2, 0.0, -np.pi/2, np.pi/2]
        # self.a = a
        # self.d = d
        # self.theta = np.array([np.pi/2, np.pi/2, 0.0, -np.pi/2, 0.0, -np.pi/2, 0.0])
        # self.theta_t = theta_t

        # 四自由度
        self.alpha = [-np.pi/2, np.pi/2, 0.0, 0.0]
        self.a = [0, 0, 0.315, 0.295]
        self.d = [0.159, -0.1255, 0.0, 0.0]
        self.theta = np.array([np.pi/2, 0.0, 0.0, 0.0])
        self.theta_t = theta_t
        
        for i in range(len(self.alpha)):
            self.theta[i] = self.theta[i] + self.theta_t[i]

        pass

    def EveryAxisTransfMatix(self):
        # 计算相邻关节变换矩阵T_i_i+1
        Ti_i1 = []
        for i in range(len(self.alpha)):
            Ti = np.array([[cos(self.theta[i]), -sin(self.theta[i]), 0, self.a[i]],
                            [sin(self.theta[i])*cos(self.alpha[i]), cos(self.theta[i])*cos(self.alpha[i]), -sin(self.alpha[i]), -sin(self.alpha[i])*self.d[i]],
                            [sin(self.theta[i])*sin(self.alpha[i]), cos(self.theta[i])*sin(self.alpha[i]), cos(self.alpha[i]), cos(self.alpha[i])*self.d[i]],
                            [0, 0, 0, 1]])
            Ti_i1.append(Ti)
        
        # 计算各坐标系相对于基坐标系的变换矩阵T
        T = []
        Ttmp = np.eye(4)
        for i in range(len(self.alpha)):
            Ttmp = Ttmp@Ti_i1[i]
            T.append(Ttmp)

        return T
    
    def TransfMatix(self):
        # 计算相邻关节变换矩阵T_i_i+1
        Ti_i1 = []
        for i in range(len(self.alpha)):
            Ti = np.array([[cos(self.theta[i]), -sin(self.theta[i]), 0, self.a[i]],
                            [sin(self.theta[i])*cos(self.alpha[i]), cos(self.theta[i])*cos(self.alpha[i]), -sin(self.alpha[i]), -sin(self.alpha[i])*self.d[i]],
                            [sin(self.theta[i])*sin(self.alpha[i]), cos(self.theta[i])*sin(self.alpha[i]), cos(self.alpha[i]), cos(self.alpha[i])*self.d[i]],
                            [0, 0, 0, 1]])
            # print(Ti)
            # print(i)
            # print("="*50)
            Ti_i1.append(Ti)
        
        # 计算各坐标系相对于基坐标系的变换矩阵T
        T = np.eye(4)
        for i in range(len(self.alpha)):
            T = T@Ti_i1[i]

        T = np.asarray(T)
        return T
    
    def Jacobian(self):
        T = self.EveryAxisTransfMatix()
        T = np.asarray(T)

        z = []
        p = []
        for i in range(len(self.alpha)):
            Ti = T[i]
            zi = Ti[0:3, 2]
            pi = Ti[0:3, 3]

            z.append(zi)
            p.append(pi)

        # 各关节雅克比矩阵组合
        Jt = []
        for i in range(len(self.alpha)):
            Jvi = np.cross(z[i], p[-1]-p[i]).reshape(-1, 1)
            Jwi = z[i].reshape(-1, 1)
            Ji = np.vstack((Jvi, Jwi))

            Jt.append(Ji)

        # 求解关节空间到末端姿态的雅克比矩阵
        for i in range(len(self.alpha)):
            if i==0:
                J = np.hstack((Jt[i],Jt[i+1]))
            elif i>1:
                J = np.hstack((J,Jt[i]))

        return J
    
    def FK(self):
        J = self.Jacobian()

        V = J@self.dtheta

        return V
    
if __name__ == "__main__":
    # theta = np.array([np.pi/50, np.pi/50, 0.0, np.pi/50, np.pi/50, 0.0, 0.0])
    # theta = np.array([0.0, 0.8*np.pi, 0.2*np.pi, 0.0])
    theta = np.array([0.0, 0.0, 0.0, 0.0])
    dtheta = np.array([0, 1.0, 0.0, 0])
    T = ForwardKinematics(theta).TransfMatix()
    J = ForwardKinematics(theta).Jacobian()
    T = np.round(T, 4)
    R_matrix = T[0:3,0:3]
    # 创建旋转对象
    rotation = R.from_matrix(R_matrix)

    # 将旋转矩阵转换为欧拉角，使用ZYX顺序
    euler_angles = rotation.as_euler('zyx')
    euler_angles[1] = euler_angles[1] - np.pi/2

    euler_angles = np.round(euler_angles, 4)
    J = np.round(J, 4)
    print(T)
    print(J)
    print(np.linalg.pinv(J))
    print(euler_angles)
    print(J@dtheta.reshape(4,1))