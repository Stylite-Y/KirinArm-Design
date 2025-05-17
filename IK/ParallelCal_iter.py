"""
腕部并联机构正逆运动学
"""

import numpy as np
from numpy import sin as s
from numpy import cos as c

# 基于坐标变换计算并联机构逆运动学，数值迭代计算正运动学
class ParallelNumCal:
    def __init__(self):
        self.a = 0.02           # 电机输出端转臂长度
        self.h = 0.04           # 两电机间距与固定坐标系的原点的距离
        self.d = 0.081          # 并联机构腕部横杆长度
        self.b = 0.004          # 横杆与腕部旋转坐标系原点垂直z方向距离
        self.H = 0.174          # 固定坐标系的原点与腕部旋转坐标系原点距离
        self.L1 = 0.218         # 电机5的长杆长度
        self.L2 = 0.138         # 电机6的长杆长度
        pass
    
    # 姿态角,电机角度
    def JacobianCal(self, p, q_real):
        a = self.a
        d = self.d
        b = self.b
        B1_0 = np.array([-a*c(p[1]) + 0.5*d*s(p[0])*s(p[1]) + b*c(p[0])*s(p[1]),
                        0.5*d*c(p[0]) - b*s(p[0]),
                        a*s(p[1]) + 0.5*d*s(p[0])*c(p[1]) + b*c(p[0])*c(p[1]) + self.H])
        
        B2_0 = np.array([-a*c(p[1]) - 0.5*d*s(p[0])*s(p[1]) + b*c(p[0])*s(p[1]),
                        -0.5*d*c(p[0]) - b*s(p[0]),
                        a*s(p[1]) - 0.5*d*s(p[0])*c(p[1]) + b*c(p[0])*c(p[1]) + self.H])
        
        # 固定坐标系下的长杆电机端坐标
        A1_0 = np.array([-a*c(q_real[0]), 0.5*d, -self.h + a*s(q_real[0])])
        A2_0 = np.array([-a*c(q_real[1]), -0.5*d, self.h + a*s(q_real[1])])

        dx1 = B1_0 - A1_0
        dx2 = B2_0 - A2_0

        # 计算雅可比矩阵
        J11 = (dx1[0]*(0.5*d*c(p[0])*s(p[1])-b*s(p[0])*s(p[1])) + dx1[1]*(-0.5*d*s(p[0])-b*c(p[0])) + \
            dx1[2]*(0.5*d*c(p[0])*c(p[1])-b*s(p[0])*c(p[1])))/(dx1[0]*a*s(q_real[0])+dx1[2]*a*c(q_real[0]))
        J12 = (dx1[0]*(a*s(p[1])+0.5*d*s(p[0])*c(p[1])+b*c(p[0])*c(p[1])) + dx1[1]*0 + \
            dx1[2]*(a*c(p[1])-0.5*d*s(p[0])*s(p[1])-b*c(p[0])*s(p[1])))/(dx1[0]*a*s(q_real[0])+dx1[2]*a*c(q_real[0]))
        J21 = (dx2[0]*(-0.5*d*c(p[0])*s(p[1])-b*s(p[0])*s(p[1])) + dx2[1]*(0.5*d*s(p[0])-b*c(p[0])) + \
            dx2[2]*(-0.5*d*c(p[0])*c(p[1])-b*s(p[0])*c(p[1])))/(dx2[0]*a*s(q_real[1])+dx2[2]*a*c(q_real[1]))
        J22 = (dx2[0]*(a*s(p[1])-0.5*d*s(p[0])*c(p[1])+b*c(p[0])*c(p[1])) + dx2[1]*0 + \
            dx2[2]*(a*c(p[1])+0.5*d*s(p[0])*s(p[1])-b*c(p[0])*s(p[1])))/(dx2[0]*a*s(q_real[0])+dx2[2]*a*c(q_real[0]))
        
        Jac = np.array([[J11, J12], [J21, J22]])

        return Jac

    # 通过末端pitch和roll方向的角度phi和psi计算两个电机输出端的旋转角度alpha
    def EulerToMotorAngle(self, psi, phi):
        # 腕部旋转坐标系与固定坐标系的其次变换矩阵
        T_01 = np.array([[np.cos(phi), np.sin(phi)*np.sin(psi), np.sin(phi)*np.cos(psi), 0],
                         [0, np.cos(psi), -np.sin(psi), 0],
                         [-np.sin(phi), np.cos(phi)*np.sin(psi), np.cos(phi)*np.cos(psi), self.H],
                         [0, 0, 0, 1]])
        
        
        # 旋转坐标系下的长杆腕部端坐标
        B1_1= np.array([-self.a, self.d/2, self.b, 1])
        B2_1= np.array([-self.a, -self.d/2, self.b, 1])

        # 固定坐标系下的长杆腕部端坐标
        B1_0 = T_01 @ B1_1
        B2_0 = T_01 @ B2_1

        _B1_0 = B1_0[0:3] - np.array([0, self.d/2, -self.h])
        _B2_0 = B2_0[0:3] - np.array([0, -self.d/2, self.h])

        # 电机5角度计算
        Den_1 = np.sqrt( _B1_0[0]**2+ _B1_0[2]**2)
        angle_tmp_1 = np.arctan(_B1_0[0]/_B1_0[2])
        alpha_1 = np.arcsin(-(self.L1**2 - np.sum(_B1_0**2) - self.a**2)/(2*self.a*Den_1)) + angle_tmp_1

        # 电机6角度计算
        Den_2 = np.sqrt( _B2_0[0]**2+ _B2_0[2]**2)
        angle_tmp_2 = np.arctan(_B2_0[0]/_B2_0[2])
        alpha_2 = np.arcsin(-(self.L2**2 - np.sum(_B2_0**2) - self.a**2)/(2*self.a*Den_2)) + angle_tmp_2

        # 电机角度：[电机5角度， 电机6角度]
        alpha = np.array([alpha_1, alpha_2])

        # print(T_01)
        # print(B1_1, B2_1)
        # print(B1_0, B2_0)
        # print(_B1_0, _B2_0)
        # print(Den_1, angle_tmp_1, alpha_1)

        return alpha

    # 通过两个电机输出端的旋转角度alpha计算末端pitch和roll方向的角度phi和psi：迭代法
    def MotorAngleToEuler(self, q_ref):
        a = self.a
        d = self.d
        b = self.b
        Numiter = 20

        # 迭代的末端pitch和roll角
        i=0
        # 初始姿态角设置
        p = np.array([0,0])
        # 初始误差设置
        q_err = np.array([1, 1])
        while np.abs(q_err[0]) > 1e-4 or np.abs(q_err[1]) > 1e-4:
            # 计算当前末端姿态下电机角度
            q_real = self.EulerToMotorAngle(p[0], p[1])
            # print("q_real: ", q_real)
            # print("="*50)
            # print("theta_now: ", p)
            # 固定坐标系下的长杆腕部端坐标
            Jac = self.JacobianCal(p, q_real)

            # 计算当前误差
            q_err = q_real - q_ref

            # 迭代法计算末端姿态角
            p = p - np.linalg.inv(Jac) @ q_err
            print("Jac: ", Jac)
            print("q_real: ", q_real)
            print("q_err: ", q_err)
            print("theta_next: ", p)
            print(i)

            # 最大迭代步数
            if i > Numiter:
                break
            i+=1

            print("="*50)
        
        # 末端姿态角：[roll角度，pitch角度]
        theta  = p
        return theta


if __name__ == "__main__":
    ParaCal = ParallelNumCal()
    endang = np.array([20/180*np.pi, 10/180*np.pi])

    # 通过末端pitch和roll方向的角度phi和psi计算两个电机输出端的旋转角度alpha
    alpha = ParaCal.EulerToMotorAngle(endang[0], endang[1])       # 依次是roll角度psi，pitch角度phi
    print("="*50)

    # 通过两个电机输出端的旋转角度alpha计算末端pitch和roll方向的角度phi和psi：迭代法
    theta = ParaCal.MotorAngleToEuler(alpha)

    print("*"*50)
    print("电机角度: ", alpha/np.pi*180)                                 # 依次时电机5和电机6的角度
    print("末端姿态角_参考: ", endang)                # 依次是roll角度，pitch角度
    print("末端姿态角_计算: ", theta)
    
