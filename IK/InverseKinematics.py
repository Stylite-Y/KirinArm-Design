"""
author: Yanyan Yuan
date: 2023.11.23
discription: this scripts is to solve IK using Jacobian-method: PseudoInverse, TransposeJacobian, damping least squares(DLS) method
input params: 
        desired_state: Desired pos homogeneous transform (4x4)
        theta_current: current theta of 7 dof joints (7x1)
        mode: solver selection: must be on of mode = ["PseudoInverse", "TransposeJacobian", "DampLS"]
output: The joint angle corresponding to the target posture (7x1)

2024.12.21:
    添加通过ikpy求解库求解四自由度机械臂逆运动学, 耗时大约45ms
2024.12.22:
    添加通过dkl求解库求解四自由度机械臂逆运动学, 耗时大约5ms,但是会容易收敛到边界解
2024.12.23:
    添加通过tracik求解库求解四自由度机械臂逆运动学, 耗时大约0.5ms,求解结果比较好
"""

import sys
import numpy as np
import time
sys.path.append('../../IK')
from ForwardKinematics import ForwardKinematics
from spatialmath import SE3, SO3
from spatialmath.base import trnorm
import warnings
import logging
import PyKDL as kdl
from tracikpy import TracIKSolver
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

from scipy.spatial.transform import Rotation as R

class InverseKinematics():
    def __init__(self, desired_state, theta_current, mode = "PseudoInverse", LogFlag = False):
        self.desired_state = desired_state
        self.state_target = self.desired_state
        self.mode = mode
        self.theta_current = theta_current
        self.err_max = 1e-3
        self.iteration_max = 50
        self.IK_seg_iternum = 1
        self.LogFlag = LogFlag
        # 配置日志输出的格式
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        # 创建Logger实例
        self.logger = logging.getLogger(__name__)

        assert desired_state.shape==(4,4), "desired_state must be (4x4) homogeneous transform matrix"
        assert theta_current.shape[0]==7 or theta_current.shape[0]==4, "theta_current must be (7x1) or (47x1) matrix"
        pass

    def getDeltaState(self, current_state, desired_state):
        R_current = current_state[0:3, 0:3]
        R_des = desired_state[0:3, 0:3]
        delta_R = R_des @ np.linalg.inv(R_current)
        delta_R_vec = SO3(trnorm(delta_R)).log(twist=True)
        delta_S_vec = np.array([desired_state[0,3]-current_state[0,3],desired_state[1,3]-current_state[1,3], desired_state[2,3]-current_state[2,3],
                                delta_R_vec[0], delta_R_vec[1], delta_R_vec[2]])

        return delta_S_vec
    
    @staticmethod
    def getStateVecByEuler(T):
        Rotation = T[0:3, 0:3]
        r = R.from_matrix(Rotation)
        euler_angle = r.as_euler('xyz')
        Vec = np.array([T[0,3],T[1,3], T[2,3],euler_angle[0], euler_angle[1], euler_angle[2]])
        
        return Vec

    def CalSegmentTrandMatrix(self, i):
        # 从当前关节角构建变换矩阵T
        State_current = ForwardKinematics(self.theta_current).TransfMatix()
        Vec_current = self.getStateVecByEuler(State_current)
        
        # 从T矩阵分离出位置信息和姿态信息
        Pos_current = Vec_current[0:3]
        angle_current = Vec_current[3:6]
        
        Vec_desired = self.getStateVecByEuler(self.state_target)
        # 从T矩阵分离出位置信息和姿态信息
        Pos_desired = Vec_desired[0:3]
        angle_desired = Vec_desired[3:6]
        
        # 计算期望位置和当前位置差值，然后分段差值
        pos_delta = Pos_desired - Pos_current
        angle_delta = angle_desired - angle_current
        euler_angle = angle_current + i*angle_delta/self.IK_seg_iternum

        # 将欧拉角转换为旋转矩阵
        r = R.from_euler('xyz', euler_angle)
        RotationM = r.as_matrix()

        # 更新分段后得到的新的变换矩阵T
        Pos_ref = Pos_current + i*pos_delta/self.IK_seg_iternum
        Pos_ref = Pos_ref.reshape(-1, 1)

        TransMatrix_ref = np.hstack((RotationM, Pos_ref))
        TransMatrix_ref = np.vstack((TransMatrix_ref, np.array([[0, 0, 0, 1]])))

        # if i == 1:
        #     print(Vec_current)
        #     print(Vec_desired)
        #     print(Pos_ref, euler_angle)

        return TransMatrix_ref

    def logprint(self, data):
        table = """
        ------------------------------------------------------------------------------------
        |    iteration       | {iteration}                                                           |
        |    error_now       | {error_now}                                                    |
        |    error_next      | {error_next}                                                  |
        |    theta_now       | {theta_now}   |
        |    dtheta          | {dtheta}   |
        |    theta_next      | {theta_next}   |
        ------------------------------------------------------------------------------------
        """
        info = {
            "iteration": data[0],
            "error_now": np.round(data[1], 6),
            "error_next": np.round(data[2], 8),
            "theta_now": data[3],
            "dtheta": data[4],
            "theta_next": data[5]
        }
        self.logger.info(table.format(**info))

    def IK(self):
        # print("the IK solver is ", self.mode)

        # 若target过大，进行分段求解
        S_current = ForwardKinematics(self.theta_current).TransfMatix()
        delta_S_vec = self.getDeltaState(S_current, self.desired_state)
        delta_num_round = delta_S_vec
        for i in range(len(delta_S_vec)):
            if i < 3:
                delta_num_round[i] = np.floor(delta_S_vec[i] / 0.02)
            else:
                delta_num_round[i] = np.floor(delta_S_vec[i] / 0.05)

        segnum = max(delta_num_round)

        # self.IK_seg_iternum = max(segnum, self.IK_seg_iternum)

        if self.IK_seg_iternum <=1:
            if self.mode == "PseudoInverse":
                # while seg_iternum < self.IK_seg_iternum:
                #     theta_desired = self.PseudoInverse()
                print("----------Method: PseudoInverse------------")
                theta_desired = self.PseudoInverse()

            elif self.mode == "TransposeJacobian":
                print("----------Method: TransposeJacobian------------")
                theta_desired = self.TransposeJacobian()
            elif self.mode == "DampLS":
                print("----------Method: DampLS------------")
                theta_desired = self.DampLS()
            else:
                assert False, "Mode ERROR: mode must be on of PseudoInverse, TransposeJacobian, DampLS"
        else:
            # print("segment:", self.IK_seg_iternum, segnum)
            # self.IK_seg_iternum = 100
            i = 1
            t0 = time.time()
            while i < self.IK_seg_iternum+1:
                t1 = time.time()
                self.desired_state = self.CalSegmentTrandMatrix(i)
                
                # theta_calc = self.PseudoInverse()
                theta_calc = self.DampLS()

                self.theta_current = theta_calc
                t2 = time.time()
                i+=1
            pass
        # theta_desired = theta_desired

        return theta_desired
    
    ## IK solution based on PseudoInverse method
    def PseudoInverse(self):
        iternum = 0
        theta_now = self.theta_current

        S_current = ForwardKinematics(theta_now).TransfMatix()

        ## end State vector get by rotation-to-euler
        # Vec_desired = self.getStateVecByEuler(self.desired_state)
        # Vec_current = self.getStateVecByEuler(S_current)
        # delta_S_vec = Vec_desired - Vec_current

        ## end State vector get by SO3 Lie group transform
        delta_S_vec = self.getDeltaState(S_current, self.desired_state)
        
        delta = np.linalg.norm(delta_S_vec)
        dtheta_old = 0

        while delta > self.err_max and iternum < self.iteration_max:
            iternum += 1

            Jacob = ForwardKinematics(theta_now).Jacobian()

            Jacb_psedinv = Jacob.T @ np.linalg.pinv(Jacob @ Jacob.T)
            dtheta = Jacb_psedinv @ delta_S_vec

            # dtheta = np.linalg.pinv(Jacob) @ delta_S_vec
            # dtheta = Jacob.T @ delta_S_vec

            theta_iter = theta_now + dtheta

            S_current_new = ForwardKinematics(theta_iter).TransfMatix()
            delta_S_vec_new = self.getDeltaState(S_current_new, self.desired_state)
            delta_new = np.linalg.norm(delta_S_vec_new)
            
            if delta_new <= delta:
                if iternum % 1 ==0 and self.LogFlag:
                    self.logprint([iternum, delta, delta_new, theta_now, dtheta, theta_iter])
                dtheta_old = dtheta
                S_current = S_current_new 
                delta_S_vec = delta_S_vec_new
                delta = delta_new
                theta_now = theta_iter
            else:
                dtheta = dtheta_old
                theta_iter = theta_now + dtheta
                S_current_new = ForwardKinematics(theta_iter).TransfMatix()
                delta_S_vec_new = self.getDeltaState(S_current_new, self.desired_state)
                delta_new = np.linalg.norm(delta_S_vec_new)                
                if iternum % 1 ==0 and self.LogFlag:
                    print("------last dtheta-------")
                    self.logprint([iternum, delta, delta_new, theta_now, dtheta, theta_iter])

                # if delta_new <= delta:
                dtheta_old = dtheta
                S_current = S_current_new 
                delta_S_vec = delta_S_vec_new
                delta = delta_new
                theta_now = theta_iter
        # print(iternum)
        if iternum == self.iteration_max:
            warnings.warn("The iteration steps has reached the maximum, The results may not meet the accuracy requirements!")

        return theta_now
    
    ## IK solution based on TransposeJacobian method
    def TransposeJacobian(self):
        iternum = 0
        
        theta_now = self.theta_current
        S_current = ForwardKinematics(theta_now).TransfMatix()

        ## end State vector get by rotation-to-euler
        # Vec_desired = self.getStateVecByEuler(self.desired_state)
        # Vec_current = self.getStateVecByEuler(S_current)
        # delta_S_vec = Vec_desired - Vec_current

        ## end State vector get by SO3 Lie group transform
        delta_S_vec = self.getDeltaState(S_current, self.desired_state)
        
        delta = np.linalg.norm(delta_S_vec)
        dtheta_old = 0

        while delta > self.err_max and iternum < self.iteration_max:
            iternum += 1

            # delta_S_vec = self.getDeltaState(S_current, self.desired_state)

            Jacob = ForwardKinematics(theta_now).Jacobian()
            
            # 计算系数alpha
            jjte = Jacob @ Jacob.T @ delta_S_vec
            alpha = np.dot(delta_S_vec, jjte) / np.dot(jjte, jjte)
            # alpha = 0.01

            dtheta = alpha*Jacob.T@delta_S_vec

            theta_iter = theta_now + dtheta

            S_current_new = ForwardKinematics(theta_iter).TransfMatix()
            delta_S_vec_new = self.getDeltaState(S_current_new, self.desired_state)
            delta_new = np.linalg.norm(delta_S_vec_new)                
            
            if delta_new <= delta:
                if iternum % 1 ==0 and self.LogFlag:
                    self.logprint([iternum, delta, delta_new, theta_now, dtheta, theta_iter])
                dtheta_old = dtheta
                S_current = S_current_new 
                delta_S_vec = delta_S_vec_new
                delta = delta_new
                theta_now = theta_iter
            else:
                dtheta = dtheta_old
                theta_iter = theta_now + dtheta
                S_current_new = ForwardKinematics(theta_iter).TransfMatix()
                delta_S_vec_new = self.getDeltaState(S_current_new, self.desired_state)
                delta_new = np.linalg.norm(delta_S_vec_new)                
                if iternum % 1 ==0 and self.LogFlag:
                    print("------last dtheta-------")
                    self.logprint([iternum, delta, delta_new, theta_now, dtheta, theta_iter])

                # if delta_new <= delta:
                dtheta_old = dtheta
                S_current = S_current_new 
                delta_S_vec = delta_S_vec_new
                delta = delta_new
                theta_now = theta_iter

        if iternum == self.iteration_max:
            warnings.warn("The iteration steps has reached the maximum, The results may not meet the accuracy requirements!")

        return theta_now
    
    ## IK solution based on damping least squares method
    def DampLS(self):
        iternum = 0
        theta_now = self.theta_current
        damping = 0.006

        S_current = ForwardKinematics(theta_now).TransfMatix()

        # Vec_desired = self.getStateVecByEuler(self.desired_state)
        # Vec_current = self.getStateVecByEuler(S_current)
        # delta_S_vec = Vec_desired - Vec_current

        delta_S_vec = self.getDeltaState(S_current, self.desired_state)
        
        delta = np.linalg.norm(delta_S_vec)
        dtheta_old = 0

        while delta > self.err_max and iternum < self.iteration_max:
            iternum += 1

            Jacob = ForwardKinematics(theta_now).Jacobian()
            Jacob_LS = np.linalg.inv(Jacob@Jacob.T + damping**2*np.eye(6))
            dtheta = Jacob.T @ Jacob_LS @ delta_S_vec

            theta_iter = theta_now + dtheta

            S_current_new = ForwardKinematics(theta_iter).TransfMatix()
            delta_S_vec_new = self.getDeltaState(S_current_new, self.desired_state)
            delta_new = np.linalg.norm(delta_S_vec_new)                
            
            if delta_new <= delta:
                if iternum % 1 ==0 and self.LogFlag:
                    self.logprint([iternum, delta, delta_new, theta_now, dtheta, theta_iter])
                dtheta_old = dtheta
                S_current = S_current_new 
                delta_S_vec = delta_S_vec_new
                delta = delta_new
                theta_now = theta_iter
            else:
                dtheta = dtheta_old
                theta_iter = theta_now + dtheta
                S_current_new = ForwardKinematics(theta_iter).TransfMatix()
                delta_S_vec_new = self.getDeltaState(S_current_new, self.desired_state)
                delta_new = np.linalg.norm(delta_S_vec_new)                
                if iternum % 1 ==0 and self.LogFlag:
                    print("------last dtheta-------")
                    self.logprint([iternum, delta, delta_new, theta_now, dtheta, theta_iter])

                # if delta_new <= delta:
                dtheta_old = dtheta
                S_current = S_current_new 
                delta_S_vec = delta_S_vec_new
                delta = delta_new
                theta_now = theta_iter
            # print(delta_new)
        if iternum == self.iteration_max:
            warnings.warn("The iteration steps has reached the maximum, The results may not meet the accuracy requirements!")

        return theta_now

# 通过ikpy计算逆运动学
class IKpy:
    def __init__(self):
        self.chain = self._chain_define()

    def _chain_define(self):
        my_chain = Chain(name='4dof_arm', active_links_mask=[False, True, True, True, True, False],
            links=[
            OriginLink(),
            URDFLink(
                name="joint1",
                bounds=(-0.78, 3.1),
                origin_translation=[0, 0, 0.0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0]
            ),
            URDFLink(
                name="joint2",
                bounds=(-1.57, 4.8),
                origin_translation=[0, 0.1575, 0],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0]
            ),
            URDFLink(
                name="joint3",
                bounds=(0.3, 2.0),
                origin_translation=[0.1255, 0, -0.315],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0]
            ),
            URDFLink(
                name="joint4",
                bounds=(-1.3, 1.3),
                origin_translation=[0, 0, -0.292],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 0]
            ),
            URDFLink(
                name="racket",
                bounds=(0.0, 0.0),
                origin_translation=[0, 0, -0.6],
                origin_orientation=[0, 0, 0],
                joint_type = "fixed",
            ),
        ])

        return my_chain
    
    def solve_ik(self, target_pos, target_ori, initial):
        inverse_kinematics = self.chain.inverse_kinematics(target_position=target_pos,
                                                           target_orientation=target_ori, 
                                                           orientation_mode="all", 
                                                           initial_position = initial)
        return inverse_kinematics   

    def _fk(self, angle):
        res = self.chain.forward_kinematics(angle)

        pos = res[0:3, 3]
        rot = res[0:3, 0:3]

        euler = R.from_matrix(rot)
        euler_ang = euler.as_euler('xyz')

        return pos, euler_ang

# 通过pykdl求解逆运动学
class PyKDL:
    def __init__(self):
        self.chain = self.create_chain()
        print("ik init done")

        pass

    def create_chain(self):
        # 创建一个新的链条
        chain = kdl.Chain()

        # 添加关节和连杆到链条
        # 假设设置了四个 revolute joints（旋转关节）和相应的链段
        
        # Joint 1
        joint1 = kdl.Joint(kdl.Joint.RotY)
        frame1 = kdl.Frame(kdl.Vector(0.0, 1.0, 0.0))  # 连杆长度、方向等
        segment1 = kdl.Segment(joint1, frame1)
        chain.addSegment(segment1)

        # Joint 2
        joint2 = kdl.Joint(kdl.Joint.RotX)
        frame2 = kdl.Frame(kdl.Vector(0.0, 0.0, -1.0))
        segment2 = kdl.Segment(joint2, frame2)
        chain.addSegment(segment2)

        # Joint 3
        joint3 = kdl.Joint(kdl.Joint.RotX)
        frame3 = kdl.Frame(kdl.Vector(-1.0, 0.0, -1.0))
        segment3 = kdl.Segment(joint3, frame3)
        chain.addSegment(segment3)

        # Joint 4
        joint4 = kdl.Joint(kdl.Joint.RotX)
        frame4 = kdl.Frame(kdl.Vector(0.0, 0.0, -1.0))
        segment4 = kdl.Segment(joint4, frame4)
        chain.addSegment(segment4)

        return chain
    
    def create_joint_limits(self):
        """
        创建关节限制
        返回: (最小角度数组, 最大角度数组)
        """
        # 设置各关节的角度限制（弧度）
        joint_limits_min = kdl.JntArray(4)
        joint_limits_max = kdl.JntArray(4)
        
        # 设置每个关节的限制范围
        # 示例：关节1-4的限制范围分别为：
        # 关节1: -180° 到 +180°
        # 关节2: -90° 到 +90°
        # 关节3: -120° 到 +120°
        # 关节4: -180° 到 +180°
        joint_limits_min[0] = np.radians(-45)  # 第1个关节最小角度
        joint_limits_min[1] = np.radians(-90)   # 第2个关节最小角度
        joint_limits_min[2] = np.radians(0.0)  # 第3个关节最小角度
        joint_limits_min[3] = np.radians(-90)  # 第4个关节最小角度
        
        joint_limits_max[0] = np.radians(180)   # 第1个关节最大角度
        joint_limits_max[1] = np.radians(270)    # 第2个关节最大角度
        joint_limits_max[2] = np.radians(120)   # 第3个关节最大角度
        joint_limits_max[3] = np.radians(90)   # 第4个关节最大角度
        
        return joint_limits_min, joint_limits_max

    def solve_ik_with_limits(self, target_pos, target_rot, initial_guess=None):
        """
        带关节限制的逆运动学求解
        
        参数:
        chain: KDL运动链
        target_pos: 目标位置 [x, y, z]
        target_rot: 目标旋转(RPY角度) [r, p, y]
        initial_guess: 初始关节角度猜测值
        
        返回:
        success: 是否成功求解
        joint_angles: 关节角度解
        """
        # 创建求解器
        fk = kdl.ChainFkSolverPos_recursive(self.chain)
        ik_v = kdl.ChainIkSolverVel_pinv(self.chain)
        
        # 创建关节限制
        joint_limits_min, joint_limits_max = self.create_joint_limits()
        
        # 创建带关节限制的求解器
        ik = kdl.ChainIkSolverPos_NR_JL(
            self.chain,
            joint_limits_min,
            joint_limits_max,
            fk,
            ik_v,
            maxiter=100, 
            eps=1e-6
        )
        
        # 设置目标位姿
        target_frame = kdl.Frame(
            kdl.Rotation(target_rot[0][0], target_rot[0][1], target_rot[0][2],
                         target_rot[1][0], target_rot[1][1], target_rot[1][2],
                         target_rot[2][0], target_rot[2][1], target_rot[2][2]),
            kdl.Vector(target_pos[0], target_pos[1], target_pos[2])
        )
        
        # 设置初始关节角度
        if initial_guess is None:
            initial_guess = [0.0] * self.chain.getNrOfJoints()
        joint_init = kdl.JntArray(self.chain.getNrOfJoints())
        for i in range(self.chain.getNrOfJoints()):
            joint_init[i] = initial_guess[i]
        
        # 求解结果
        joint_angles = kdl.JntArray(self.chain.getNrOfJoints())
        
        # 进行求解
        result = ik.CartToJnt(joint_init, target_frame, joint_angles)
        
        # 获取结果
        success = (result >= 0)
        angles = [joint_angles[i] for i in range(self.chain.getNrOfJoints())]
        print("angle: ", angles)
        
        return success, angles

    def solve_ik(self, target_pos, target_rot, initial_guess=None):
        """
        求解逆运动学
        
        参数:
        chain: KDL运动链
        target_pos: 目标位置 [x, y, z]
        target_rot: 目标旋转(RPY角度) [r, p, y]
        initial_guess: 初始关节角度猜测值
        
        返回:
        success: 是否成功求解
        joint_angles: 关节角度解
        """
        # 创建求解器
        print("ik sovler")
        fk = kdl.ChainFkSolverPos_recursive(self.chain)
        ik_v = kdl.ChainIkSolverVel_pinv(self.chain)
        print("ik 1")
        ik = kdl.ChainIkSolverPos_NR(self.chain, fk, ik_v, maxiter=100)
        print("ik 2")
        
        # 设置目标位姿
        target_frame = kdl.Frame(
            kdl.Rotation.RPY(target_rot[1], target_rot[0], target_rot[2]),
            kdl.Vector(target_pos[0], target_pos[1], target_pos[2])
        )
        print("ik 3")
        
        # 初始关节角度
        if initial_guess is None:
            initial_guess = [0.0] * self.chain.getNrOfJoints()
        joint_init = kdl.JntArray(self.chain.getNrOfJoints())
        for i in range(self.chain.getNrOfJoints()):
            joint_init[i] = initial_guess[i]
        print("ik 4: ", self.chain.getNrOfJoints())
        
        # 求解结果
        joint_angles = kdl.JntArray(self.chain.getNrOfJoints())
        print("ik 5")
        
        # 进行求解
        result = ik.CartToJnt(joint_init, target_frame, joint_angles)
        print("ik 6")
        
        # 返回结果
        success = (result >= 0)
        angles = [joint_angles[i] for i in range(self.chain.getNrOfJoints())]
        
        return success, angles

    # 计算雅可比矩阵
    def compute_jacobian(self, joint_angles):
        q = kdl.JntArray(len(joint_angles))
        for i in range(len(joint_angles)):
            q[i] = joint_angles[i]

        # 获得雅可比矩阵
        # Create a KDL Jacobian
        jacobian = kdl.Jacobian(self.chain.getNrOfJoints())

        # Create a Jacobian solver
        jac_solver = kdl.ChainJntToJacSolver(self.chain)
        # Compute the Jacobian
        jac_solver.JntToJac(q, jacobian)
        jac_npmatrix = np.zeros((jacobian.rows(), jacobian.columns()))
        for i in range(jacobian.rows()):
            for j in range(jacobian.columns()):
                jac_npmatrix[i, j] = jacobian[i, j]
        print("jacobian: ", np.around(jac_npmatrix, 4))

        return jac_npmatrix
    
# 通过trac-ik求逆运动学
class TracIk:
    def __init__(self):
        self.ik_solver = self.tracik_solver()
        pos = self.ik_solver.fk([0.0, 2.6,0.4, 0.0])
        print("num of joint: ", self.ik_solver.number_of_joints)
        print("fk test: ", pos)
        print("TracIk init done")

        pass

    def tracik_solver(self):
        # 创建一个新的求解器
        return TracIKSolver(
        "/home/yyy/Documents/BadmintonMatch/resources/KirinArm.urdf", 
        "ShoulderRoll", 
        "end_effector", 
        timeout=0.005
        )

    def solve_ik(self, target_state, initial_guess):
        res = self.ik_solver.ik(target_state.tolist(), qinit=initial_guess)

        return res



# ----------------------------------- test ----------------------------------
# 单次测试
def main():
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    theta_current = np.zeros(7)
    theta_target = np.array([0, 0, 0, np.pi/50, np.pi/50, 0, 0])
    t1 = time.time()
    state_desired = ForwardKinematics(theta_target).TransfMatix()

    state_current = ForwardKinematics(theta_current).TransfMatix()
    mode = ["PseudoInverse", "TransposeJacobian", "DampLS"]
    IK = InverseKinematics(state_desired, theta_current, mode = mode[2])
    # t1 = time.time()
    theta_calc = IK.IK()
    t2 = time.time()
    State_calc = ForwardKinematics(theta_calc).TransfMatix()
    Vec_calc = IK.getStateVecByEuler(State_calc)
    Vec_current = IK.getStateVecByEuler(state_current)
    Vec_des = IK.getStateVecByEuler(state_desired)
    print("="*50)
    print("IK solve time:", t2-t1)
    print("theta_target: ", theta_target)
    print("Pos current: ", Vec_current)
    print("Pos_target: ", Vec_des)
    print("theta_desired_calc: ", theta_calc)
    print("Pos_desired_calc: ", Vec_calc)

    return theta_calc, t2-t1

def CalSegmentTrandMatrix(theta_start, Pos_ref, euler_angle, i, theta_iternum):
    # 从当前关节角构建变换矩阵T
    State_current = ForwardKinematics(theta_start).TransfMatix()
    Vec_current = InverseKinematics.getStateVecByEuler(State_current)
       
    # 从T矩阵分离出位置信息和姿态信息
    Pos_current = np.array([Vec_current[0],Vec_current[1], Vec_current[2]])
    angle_current = np.array([Vec_current[3],Vec_current[4],Vec_current[5]])
    
    Pos_ref= Pos_ref.reshape(1, -1)
    # print(Pos_ref)
    # print(Pos_current)
    
    # 计算期望位置和当前位置差值，然后分段差值
    pos_delta = Pos_ref - Pos_current
    angle_delta = euler_angle - angle_current
    euler_angle = angle_current + i*angle_delta/theta_iternum

    # 将欧拉角转换为旋转矩阵
    r = R.from_euler('xyz', euler_angle)
    RotationM = r.as_matrix()

    # 更新分段后得到的新的变换矩阵T
    Pos_ref = Pos_current + i*pos_delta/theta_iternum
    Pos_ref = Pos_ref.reshape(-1, 1)
    # print("CalSegmentTrandMatrix")
    # print(pos_delta / theta_iternum)
    # print(Pos_ref)
    # print(angle_delta/theta_iternum)
    # print(euler_angle)
    TransMatrix_ref = np.hstack((RotationM, Pos_ref))
    TransMatrix_ref = np.vstack((TransMatrix_ref, np.array([[0, 0, 0, 1]])))
    # print(TransMatrix_ref)
    return TransMatrix_ref

# 完成轨迹测试
def main2():
    theta_start = np.zeros(7)
    # target_theta = np.array([0.0, 0.0, 0.0, np.pi/2+np.pi/6, -np.pi/6, 0.0, 0.0])

    Pos_ref = np.array([-0.1255, -0.08468, -0.04126])
    euler_angle = np.array([0, 0.0, 0.0])
    theta_iternum = 100
    mode = ["PseudoInverse", "TransposeJacobian", "DampLS"]

    i = 1
    current_theta = theta_start
    # t_all = 0
    t0 = time.time()
    while i < theta_iternum+1:
        t1 = time.time()
        state_desired = CalSegmentTrandMatrix(theta_start, Pos_ref, euler_angle, i, theta_iternum)
        IK = InverseKinematics(state_desired, current_theta, mode = mode[0])
        
        theta_calc = IK.IK()
        # Vec_desired = InverseKinematics.getStateVecByEuler(state_desired)
        # State_calc = ForwardKinematics(theta_calc).TransfMatix()
        # Vec_calc = InverseKinematics.getStateVecByEuler(State_calc)
        # State_current = ForwardKinematics(current_theta).TransfMatix()
        # Vec_current = InverseKinematics.getStateVecByEuler(State_current)

        # print("Vec_current: ", Vec_current)
        # print("Vec_desired: ", Vec_desired)
        # print("Vec_cal: ", Vec_calc)

        current_theta = theta_calc
        t2 = time.time()
        # if i == 1:
        #     break

        # print("-"*50)
        # print("IK iteration: ", i)
        # print("time: ", t2-t1)
        # t_all += t2-t1
        i+=1
        
    t_end = time.time()
    print("="*50)
    # print("all calc time: ", t_end-t0, t_all)
    # print("theta_target", target_theta)
    print("theta_calc", current_theta)

    return current_theta, t_end-t0
    pass

def main3():
    # theta_start = np.zeros(7)
    # target_theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    theta_start = np.array([0.0, 2.5, 0.6, 0.0])
    target_theta = np.array([0.0, 0.8*np.pi, 0.2*np.pi, 0.0])
    state_desired = ForwardKinematics(target_theta).TransfMatix()
    # tar = ForwardKinematics(target_theta).FK()
    # print(state_desired)
    # print(tar)

    # Pos_ref = np.array([-0.16, 0.45, -0.3])
    # euler_angle = np.array([np.pi/2, 0.0, 0.0])
    Pos_ref = np.array([-0.1255, 0.3442, 0.5498])
    euler_angle = np.array([3.14, 1.57, 0.0])
    r = R.from_euler('zyx', euler_angle)
    end_ori = r.as_matrix()
    T_target = end_ori
    T_target = np.hstack((T_target, Pos_ref.reshape(-1,1)))
    T_target = np.vstack((T_target, np.array([0,0,0,1])))

    mode = ["PseudoInverse", "TransposeJacobian", "DampLS"]
    t1 = time.time()
    IK = InverseKinematics(T_target, theta_start, mode = mode[0], LogFlag=True)
    theta_calc = IK.IK()
    t2 = time.time()

    print(theta_calc, t2-t1)

    return theta_calc, t2-t1


    # Pos_ref = np.array([-0.16, 0.45, -0.3])
    # euler_angle = np.array([np.pi/2, 0.0, 0.0])

# 测试ikpy的逆运动学结果
def ikpytest():
    target_pos = [-1, 1.5, 2.8665]
    target_ori = [np.pi, 0.0, 0.0]
    rotation = R.from_euler('xyz', target_ori)
    Rot_matrix = rotation.as_matrix()

    print(Rot_matrix)

    initial = [0.0, 0.0, np.pi/9, np.pi/9, np.pi/9, 0.0]
    
    # t1 = time.time()
    ik_solver = IKpy()
    t1 = time.time()
    res = ik_solver.solve_ik(target_pos, Rot_matrix, initial)
    t2 = time.time()

    print("time: ", t2-t1)
    print("res: ", res)


# 测试pykdl的逆运动学结果
def pykdltest():
    target_pos = [-1.0, 3, -1]
    target_ori = [np.pi/2, 0.0, 0.0]
    # target_ori = [0.0, 0.0, 0.0]
    rotation = R.from_euler('xyz', target_ori)
    Rot_matrix = rotation.as_matrix()

    print(Rot_matrix)

    initial = [0.0, np.pi*0.3, np.pi*0.1, 0.0]
    
    # t1 = time.time()
    ik_solver = PyKDL()
    t1 = time.time()
    # res = ik_solver.solve_ik(target_pos, target_ori, initial_guess=initial)
    sucess, res = ik_solver.solve_ik_with_limits(target_pos, Rot_matrix, initial_guess=initial)
    t2 = time.time()

    print("time: ", t2-t1)
    print("res: ", res)

# 测试tracik的逆运动学结果
def traciktest():
    T = np.eye(4)
    target_pos = [-0.3400, 1.031, 0.135]
    target_ori = [np.pi/2, np.pi/2, 0.0]
    # target_ori = [0.0, 0.0, 0.0]
    rotation = R.from_euler('xyz', target_ori)
    Rot_matrix = rotation.as_matrix()

    T[0:3, 3] = target_pos[0:3]
    T[0:3, 0:3] = Rot_matrix
    print(T)

    initial = [0.0, np.pi*0.1, np.pi*0.5, np.pi*0.1]
    
    # t1 = time.time()
    ik_solver = TracIk()
    t1 = time.time()
    res = ik_solver.solve_ik(T, initial_guess=initial)
    t2 = time.time()

    print("time: ", t2-t1)
    print("res: ", res)

def _inverse_kinematics():
        arm_len = [0, 1, 0.5, 0.5]
        # ball_pos = [0, np.sqrt(3)/2, 1.5]
        ball_pos = [0, -0.5, np.sqrt(3)/2]
        # ball_pos = [0, -np.sqrt(3)/2, 1.5]
        # ball_pos = [0, -3/2, np.sqrt(3)/2]
        ## 假设腕部关节此时的角度为0
        # Pos_tar = ball_pos - np.array([0.0, 0.061, 0.81])
        Pos_tar = np.array(ball_pos)
        maxlen = np.sqrt(arm_len[0]**2+
                (arm_len[1]+arm_len[2]+arm_len[3])**2)
        l_tar = np.sqrt(np.sum(Pos_tar**2))

        if l_tar>maxlen:
            Pos_tar[0] = Pos_tar[0]*(maxlen - 1e-4)/l_tar
            Pos_tar[1] = Pos_tar[1]*(maxlen - 1e-4)/l_tar
            Pos_tar[2] = Pos_tar[2]*(maxlen - 1e-4)/l_tar

        # 根据球拍位置求其theta_y的角度
        l_s = arm_len[0]
        pos = Pos_tar[0:3]
        theta = np.zeros(4)

        # 根据球拍位置求其theta_y的角度(即roll)
        tmp = l_s /np.sqrt(pos[0]**2 + pos[2]**2)
        if (abs(tmp) <= 1 and abs(pos[1] / np.sqrt(pos[0] * pos[0] + pos[2] * pos[2])) <= 1):
            theta[0] = -(np.arccos(tmp) - np.arcsin(pos[0]/np.sqrt(pos[0]**2 + pos[2]**2)) - np.pi / 2)

        # 求elbow的角度
        lr = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2 - arm_len[0]**2)
        if lr > (arm_len[1]+arm_len[2]+arm_len[3]):
            lr = arm_len[1]+arm_len[2]+arm_len[3] - 1e-4
        else:
            lr = lr
        temp = (arm_len[1]**2 + (arm_len[2]+arm_len[3])**2 - lr * lr) / \
                (2*arm_len[1]*(arm_len[2]+arm_len[3])) + 1e-5
        if (abs(temp) <= 1):
            theta[2] = np.pi - np.arccos(temp)

        # 求shoulder pitch的角度
        temp1 = pos[1]/lr
        temp2 = (lr * lr + arm_len[1]**2 - (arm_len[2]+arm_len[3])**2) / (2*lr*arm_len[1]) - 1e-5
        if (abs(temp1) <= 1 and abs(temp2) <= 1):
            theta[1] = np.pi-(np.arccos(temp2) + np.arcsin(temp1))
        
        print(np.arccos(temp2), np.arcsin(temp1))
        print(theta)

        # return theta

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    # t_s = 0
    # num = 1
    # for i in range(num):
    #     theta, dt = main3()
    #     t_s += dt
    #     print(dt)

    # state_desired = ForwardKinematics(theta).TransfMatix()
    
    # print(state_desired)
    # main3()

    # ikpytest()
    # pykdltest()
    # traciktest()
    _inverse_kinematics()
    pass

