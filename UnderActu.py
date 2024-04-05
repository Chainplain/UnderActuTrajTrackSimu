"""my_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
# from controller import Robot,Motor
from controller import Robot,Motor
from controller import Supervisor
from controller import Node
from controller import PositionSensor
from controller import Display


import numpy as np
from   Bladelement import BladeAeroCalculator as BAC
from   RotationComputation import FromPointer2Axis_Angle as P2AA
from   RotationComputation import FromRotation2Euler_Angle_in_Rad as R2EAR
from   RotationComputation import GetUnitDirection_Safe
from   ArrowManager import ForceArrowManager as FAM
from   SpecialorthogonalControl import SO_3_controller as SOC
from   RotationComputation import Angle_Trajectory_Generator as ATG
from   RotationComputation import CalcTorque_in_Robot as CTR
from   FilterLib import Low_Pass_Second_Order_Filter as LPSF
from   FTslideControl import Finite_time_slide_mode_observer_3dim as FT_observer
from   FTslideControl import Positional_Traj_Track_Controller as  PTTC
from   FTslideControl import Computing_desired_rotation
from   FTslideControl import Attitude_reference_generator as ARG
from   readTraj import traj_reader as t_reader

from   FTslideControl import Under_Traj_Track_Controller as UTT_controller
from   FTslideControl import Simplified_Att_Controller as SA_controller

from FTreducedAttCon import Reduced_Att_Controller as RAC
from datetime import datetime

import os

import random
import scipy.io as scio
# from controller import Node
# create the Robot instance.3
############### OVerall Test Region ############
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("#%Y_%m_%d_%H_%M_%S#")

Category = 'ProposedLineData'
File_name = 'Newdata/' + formatted_datetime + Category



is_filtered  = 1
##_DW_triangle_05pi
##_DW_0_0_05pi
Given_av = np.matrix([[0],[-0.5 * np.pi],[0]])
Zero_av = np.matrix([[0],[0],[0]])

T_reader = t_reader('line.mat')
# wall_obstraction.mat
# ball_obstraction_hcr_pi.mat
# line.mat
initial_translation = [0, 0, 0]
initial_rotation_in_axis_angle =  [1, 0 ,0, 0]
init_dist = 0.2
# [0, 0 ,1, np.pi/2] [1, 0 ,0, 0]

mat_file_format = '.mat'
mat_length = 10000
StrokeFreq   = 13

Total_wing_torque_list   = []
Total_wing_force_list    = []
Total_rudder_torque_list = []
Total_rudder_force_list  = []
Total_tail_torque_list   = []
Total_tail_force_list    = []

Total_body_rotation_list     = []
Total_body_translation_list  = []
Total_body_translation_desire  = []
Total_body_angular_velocity_list  = []
Total_Angular_velocity_filtered   = []
Total_desired_rotation_list     = []
Total_record_v_list = []
Total_p_d_list = []
Total_v_d_list = []
Total_psi_d_list = []


# Total_desired_translation_list  = []
Total_desired_angular_velocity_list = []
Total_psi_rotation_error_list  = []
Total_yaw_input = []
Total_roll_input = []
Total_pitch_input = []
Total_current_alt_vel_des= []
Total_current_alt_vel = []

Total_desire_gamma_list = []
Total_current_gamma_list = []

Total_Freq_stroke = []
############### OVerall Test Region ############
flapper = Supervisor()

FORCE_RELATIVECONSTANT = 0.0005
arena_Air_Density = 1.29
Simulation_Gap    = 0.001
Controller_Gap_vs_Simulation_Gap = 10
FromWing2Tail = 0.15
# dAOAfile = open("FirstTestdAOA.txt", 'a')
# dAOAfile.truncate(0)
# drawer  = Display('mirror')


LU_bac = BAC('SimpleFlapper0Wing40BLE_X_pos.npy',\
                'SimpleFlapper0Wing40BLE_Y_pos.npy',bladeElementNo=40)
LD_bac = BAC('SimpleFlapper0Wing40BLE_X_pos.npy',\
                'SimpleFlapper0Wing40BLE_Y_pos.npy',bladeElementNo=40)
RU_bac = BAC('SimpleFlapper0Wing40BLE_X_pos.npy',\
                'SimpleFlapper0Wing40BLE_Y_pos.npy',bladeElementNo=40)
RD_bac = BAC('SimpleFlapper0Wing40BLE_X_pos.npy',\
                'SimpleFlapper0Wing40BLE_Y_pos.npy',bladeElementNo=40)
rudder_bac = BAC('SimpleFlapper0Tail_V20BLE_X_pos.npy',\
                'SimpleFlapper0Tail_V20BLE_Y_pos.npy',bladeElementNo=20) 
tail_bac =   BAC('SimpleFlapper0Tail_H20BLE_X_pos.npy',\
                'SimpleFlapper0Tail_H20BLE_Y_pos.npy',bladeElementNo=20) 
                               
S_d_Flapping_wing_actuator_disk_area = np.pi * 0.15 * 0.15 * 0.5
###                                π       radius        disk half of circle


             

LU_bac.SetVirturalWingPlaneRelative2Wing([1,0,0,0,1,0,0,0,1])
LD_bac.SetVirturalWingPlaneRelative2Wing([1,0,0,0,1,0,0,0,1])
RU_bac.SetVirturalWingPlaneRelative2Wing([1,0,0,0,1,0,0,0,1])
RD_bac.SetVirturalWingPlaneRelative2Wing([1,0,0,0,1,0,0,0,1])
rudder_bac.SetVirturalWingPlaneRelative2Wing([1,0,0,0,-1,0,0,0,-1])
tail_bac.  SetVirturalWingPlaneRelative2Wing([1,0,0,0,1,0,0,0,1]) # Rot Y, -90    Rot  Z -90




R_d         = np.matrix([[1, 0, 0],\
                         [0, 1, 0],\
                         [0, 0, 1]]) 
                                       
R_z_static  = np.matrix([[1, 0, 0],\
                         [0, 1, 0],\
                         [0, 0, 1]]) 
                         
Omega_static = np.matrix([[0],\
                          [0],\
                          [0]])                          

timestep = int(flapper.getBasicTimeStep()) #/ multi_speed

 
Real_motor_LU_RD_joint  = flapper.getDevice('LU_RD_Joint')
Real_motor_LD_RU_joint  = flapper.getDevice('LD_RU_Joint')

Real_motor_LU_RD_joint_sensor = flapper.getDevice('LU_RD_Joint_sensor')
Real_motor_LD_RU_joint_sensor = flapper.getDevice('LD_RU_Joint_sensor')

Real_motor_LU_RD_joint_sensor.enable(1)
Real_motor_LD_RU_joint_sensor.enable(1)



Real_motor_LU_wing_joint = flapper.getDevice('LU_Joint')
Real_motor_LD_wing_joint = flapper.getDevice('LD_Joint')
Real_motor_RU_wing_joint = flapper.getDevice('RU_Joint')
Real_motor_RD_wing_joint = flapper.getDevice('RD_Joint')
Real_motor_rudder_joint  = flapper.getDevice('Rudder_Joint')
Real_motor_H_tail_joint  = flapper.getDevice('H_Tail_Joint')


Real_motor_LU_wing_joint_sensor = flapper.getDevice('LU_Joint_sensor')
Real_motor_LD_wing_joint_sensor = flapper.getDevice('LD_Joint_sensor')
Real_motor_RU_wing_joint_sensor = flapper.getDevice('RU_Joint_sensor')
Real_motor_RD_wing_joint_sensor = flapper.getDevice('RD_Joint_sensor')
Real_motor_rudder_joint_sensor  = flapper.getDevice('Rudder_Joint_sensor')
Real_motor_H_tail_joint_sensor  = flapper.getDevice('H_Tail_Joint_sensor')


Real_motor_LU_wing_joint_sensor.enable(1)
Real_motor_LD_wing_joint_sensor.enable(1)
Real_motor_RU_wing_joint_sensor.enable(1)
Real_motor_RD_wing_joint_sensor.enable(1)
Real_motor_rudder_joint_sensor. enable(1)
Real_motor_H_tail_joint_sensor. enable(1)

induced_velocity_record = []
induced_velocity_record_list_length = 1000
# print('Get Real_motor_LU_RD_joint position:',\
                     # Real_motor_LU_RD_joint_sensor.getValue())

# The sensor can only be read in the while loop Since only then it is sampled.

LU_wing                 = flapper.getFromDef('LU')
LD_wing                 = flapper.getFromDef('LD')
RU_wing                 = flapper.getFromDef('RU')
RD_wing                 = flapper.getFromDef('RD')
rudder                  = flapper.getFromDef('rudder')
# tail                    = flapper.getFromDef('TheFlapper') the same

TheFlapper              = flapper.getFromDef('TheFlapper')
tail                    = flapper.getFromDef('H_Tail')
Flapper_translation      = TheFlapper.getField('translation')
Flapper_rotation         = TheFlapper.getField('rotation')
# LU_RD_joint             = flapper.getFromDef('LU_RD')

### Set initial postion
# initial_translation = [0, 0, 0]
# initial_rotation_in_axis_angle = [1, 0 ,0, 0]




#'MY_ROBOT'

force = [0,0.1,0.4]

count = 0

LU_OrVec = LU_wing.getOrientation()
LD_OrVec = LD_wing.getOrientation()
RU_OrVec = RU_wing.getOrientation()
RD_OrVec = RD_wing.getOrientation()


LU_wing_Rotation_matrix = \
np.matrix([[LU_OrVec[0], LU_OrVec[1], LU_OrVec[2]],\
           [LU_OrVec[3], LU_OrVec[4], LU_OrVec[5]],\
           [LU_OrVec[6], LU_OrVec[7], LU_OrVec[8]]])
           
LD_wing_Rotation_matrix = \
np.matrix([[LD_OrVec[0], LD_OrVec[1], LD_OrVec[2]],\
           [LD_OrVec[3], LD_OrVec[4], LD_OrVec[5]],\
           [LD_OrVec[6], LD_OrVec[7], LD_OrVec[8]]])
           
RU_wing_Rotation_matrix = \
np.matrix([[RU_OrVec[0], RU_OrVec[1], RU_OrVec[2]],\
           [RU_OrVec[3], RU_OrVec[4], RU_OrVec[5]],\
           [RU_OrVec[6], RU_OrVec[7], RU_OrVec[8]]])
           
RD_wing_Rotation_matrix = \
np.matrix([[RD_OrVec[0], RD_OrVec[1], RD_OrVec[2]],\
           [RD_OrVec[3], RD_OrVec[4], RD_OrVec[5]],\
           [RD_OrVec[6], RD_OrVec[7], RD_OrVec[8]]])


LU_velVec = LU_wing.getVelocity()
LD_velVec = LD_wing.getVelocity()
RU_velVec = RU_wing.getVelocity()
RD_velVec = RD_wing.getVelocity()


Flapping_wing_induced_flow_Tail         = np.array([0, 0, 0])
# print('Velocity Vector Test Complete!\n')

vel_FreeFlow                       = np.array([0,0,0])
Flapping_wing_induced_flow         = np.array([0,0,0])
# print('LU_wing Orientation:',LU_wing.getOrientation())
# print('LD_wing Orientation:',LD_wing.getOrientation())
# print('RU_wing Orientation:',RU_wing.getOrientation())
# print('RD_wing Orientation:',RD_wing.getOrientation())
# Rudder_Comparing_set = Rudder_translation.getSFVec3f()
# Rudder_Comparing_np = np.array(Rudder_Comparing_set)
# TranslationBack2o = list(Rudder_Comparing_np + [0,0,0])
# print('TranslationBack2o:',TranslationBack2o
StrokeAmp    = np.pi / 8

StrokeAmpOffSet = StrokeAmp

# Real_motor_LU_RD_joint.maxVelocity = 100
# Real_motor_LD_RU_joint.maxVelocity = 100
CountKit = 0

Real_motor_LU_wing_joint_sensor_value = Real_motor_LU_wing_joint_sensor.getValue()
Real_motor_LD_wing_joint_sensor_value = Real_motor_LD_wing_joint_sensor.getValue()
Real_motor_RU_wing_joint_sensor_value = Real_motor_RU_wing_joint_sensor.getValue()
Real_motor_RD_wing_joint_sensor_value = Real_motor_RD_wing_joint_sensor.getValue()
Real_motor_rudder_joint_sensor_value  = Real_motor_rudder_joint_sensor.getValue()
Real_motor_H_tail_joint_sensor_value  = Real_motor_H_tail_joint_sensor.getValue()
# StrokeFreq_Smooth =  StrokeFreq
# K_StrokeFreq_Smooth = 1
Theta             = 0

###-----Position Observer-------###
Here_pos_observer = FT_observer(robot_mass = 0.02) 
Here_pos_observer. p_observer = np.mat([[initial_translation[0]],\
                                        [initial_translation[1]],\
                                        [initial_translation[2]]])


###------Controller Initialisation-------###
reduced_attitude_controller = RAC()

Inertia_matrix = np. mat( np.diag([2.29e-4, 2.43e-4, 3.17e-5]))

Gamma_now = np.mat([0,0,1]).T
Gamma_last = Gamma_now

Gamma_des_x_list_num = 11
Gamma_des_y_list_num = 11
alt_des_vel_list_num = 11

Gamma_des_x_list = np.linspace( -0.8, 0.2, Gamma_des_x_list_num)
Gamma_des_y_list = np.linspace( -0.5, 0.5, Gamma_des_y_list_num)
alt_des_vel_list = np.linspace(-0.5, 0.5, alt_des_vel_list_num)

duration_each_episode_in_sec = 15.0

alt_int = 0.0
alt_int_mag_max = 2.0
alt_err_mag_max = 4.0
alt_vel_mag_max = 3.0

alt_now = 0.0
alt_last = 0.0
alt_des = 0.0
p_d = np.mat([0,0,0]).T


###------Parameter Settings------###
roll_rudder_mag_max = 0.7
H_tail_mag_max      = 0.5

###------Iterator Settings------###
episode_i = 0

Gamma_des_x_i = np.mod(episode_i, Gamma_des_x_list_num)
Gamma_des_y_i = np.mod( int(episode_i / Gamma_des_x_list_num),  Gamma_des_y_list_num)
alt_des_vel_i = int(episode_i / Gamma_des_x_list_num / Gamma_des_y_list_num)

print('episode_', episode_i,':')
print('Gamma_des_x_i:',Gamma_des_x_i)
print('Gamma_des_y_i:',Gamma_des_y_i)
print('alt_des_vel_i:',alt_des_vel_i)

### Attitude Reference Generator
Here_ARG = ARG()
Here_ARG. generator_gap_AV = Simulation_Gap * Controller_Gap_vs_Simulation_Gap
Here_ARG. generator_gap_rotation = Simulation_Gap
## we first use stationary desire 
Flapper_Rotation_desired = np.matrix([[1,0,0],[0,1,0],[0,0,1]])

Flapper_OrVec                = TheFlapper.getOrientation()
Flapper_Rotation_matrix_initial = \
            np.matrix([[1,  0, 0],\
                       [0, 1, 0],\
                       [0, 0, 1]])

Flapper_Rotation_desired = Flapper_Rotation_matrix_initial
Flapper_Angular_velocity_desired = Given_av
Angular_velocity_filter = LPSF(Zero_av, 8, 0.8, Simulation_Gap)

Here_ATG = ATG(Flapper_Rotation_desired, Flapper_Angular_velocity_desired, Simulation_Gap * Controller_Gap_vs_Simulation_Gap)

RecordCount = 0

Controller_gap = Controller_Gap_vs_Simulation_Gap / 1000.0

Postion_Controller = UTT_controller( Controller_gap )
Postion_Controller. Control_gap = Controller_gap

Attitude_Controller = SA_controller(Controller_gap)
Attitude_Controller. Control_gap = Controller_gap

Flapper_psi_filter = LPSF(0, 8, 0.8, 0.001)


while flapper.step(timestep) != -1:
    if RecordCount == 0:
        Flapper_translation.setSFVec3f(initial_translation)
        Flapper_rotation.setSFRotation(initial_rotation_in_axis_angle)
    # print('timestep',timestep)
    
    # Now_time = flapper.getTime()
    
    # StrokeFreq_Smooth = K_StrokeFreq_Smooth * (StrokeFreq - StrokeFreq_Smooth)
    Theta = Theta +  2 * np.pi * Simulation_Gap * StrokeFreq
    # print('Theta',Theta)
    if (Theta > 2 * np.pi):
        Theta = np.mod(Theta,  2 * np.pi )
    LU_RD_joint_angle = StrokeAmp * np.sin(Theta) + StrokeAmpOffSet
    LD_RU_joint_angle = -LU_RD_joint_angle
    Real_motor_LU_RD_joint.setPosition(LU_RD_joint_angle)
    Real_motor_LD_RU_joint.setPosition(LD_RU_joint_angle)
    # print('LU_RD_joint_angle',LU_RD_joint_angle)
    # Real_motor_LU_RD_joint.setVelocity(10)
    # Real_motor_LD_RU_joint.setVelocity(10)
    # print('LU_RD_joint_angle:',LU_RD_joint_angle)

    LU_OrVec = LU_wing.getOrientation()
    LD_OrVec = LD_wing.getOrientation()
    RU_OrVec = RU_wing.getOrientation()
    RD_OrVec = RD_wing.getOrientation()
    
    rudder_OrVec = rudder.getOrientation()
    tail_OrVec   = tail.getOrientation()
    Flapper_OrVec = TheFlapper.getOrientation()
    
    LU_wing_Rotation_matrix = \
    np.matrix([[LU_OrVec[0], LU_OrVec[1], LU_OrVec[2]],\
               [LU_OrVec[3], LU_OrVec[4], LU_OrVec[5]],\
               [LU_OrVec[6], LU_OrVec[7], LU_OrVec[8]]])
               
    LD_wing_Rotation_matrix = \
    np.matrix([[LD_OrVec[0], LD_OrVec[1], LD_OrVec[2]],\
               [LD_OrVec[3], LD_OrVec[4], LD_OrVec[5]],\
               [LD_OrVec[6], LD_OrVec[7], LD_OrVec[8]]])
               
    RU_wing_Rotation_matrix = \
    np.matrix([[RU_OrVec[0], RU_OrVec[1], RU_OrVec[2]],\
               [RU_OrVec[3], RU_OrVec[4], RU_OrVec[5]],\
               [RU_OrVec[6], RU_OrVec[7], RU_OrVec[8]]])
               
    RD_wing_Rotation_matrix = \
    np.matrix([[RD_OrVec[0], RD_OrVec[1], RD_OrVec[2]],\
               [RD_OrVec[3], RD_OrVec[4], RD_OrVec[5]],\
               [RD_OrVec[6], RD_OrVec[7], RD_OrVec[8]]])
               
    rudder_Rotation_matrix = \
    np.matrix([[rudder_OrVec[0], rudder_OrVec[1], rudder_OrVec[2]],\
               [rudder_OrVec[3], rudder_OrVec[4], rudder_OrVec[5]],\
               [rudder_OrVec[6], rudder_OrVec[7], rudder_OrVec[8]]])
                   
    tail_Rotation_matrix = \
    np.matrix([[tail_OrVec[0], tail_OrVec[1], tail_OrVec[2]],\
               [tail_OrVec[3], tail_OrVec[4], tail_OrVec[5]],\
               [tail_OrVec[6], tail_OrVec[7], tail_OrVec[8]]])
               
    Flapper_Rotation_current = \
    np.matrix([[Flapper_OrVec[0], Flapper_OrVec[1], Flapper_OrVec[2]],\
               [Flapper_OrVec[3], Flapper_OrVec[4], Flapper_OrVec[5]],\
               [Flapper_OrVec[6], Flapper_OrVec[7], Flapper_OrVec[8]]])

    # LU_OrVec    = LU_wing.getOrientation()
    # LD_OrVec    = LD_wing.getOrientation()
    # RU_OrVec    = RU_wing.getOrientation()
    # RD_OrVec    = RD_wing.getOrientation()
    rudder_OrVec = rudder.getOrientation()
    
    Flapper_OrVec = TheFlapper.getOrientation()
    Flapper_OrVec_in_Euler =  R2EAR(Flapper_OrVec)
    # print('Flapper_OrVec:',Flapper_OrVec_in_Euler )
    
    # LU_RD_OrVec = LU_RD_joint.getOrientation()

    LU_bac.RequestWingOrientation2InertiaFrame (LU_OrVec)
    LD_bac.RequestWingOrientation2InertiaFrame (LD_OrVec)
    RU_bac.RequestWingOrientation2InertiaFrame (RU_OrVec)
    RD_bac.RequestWingOrientation2InertiaFrame (RD_OrVec)
    rudder_bac.RequestWingOrientation2InertiaFrame (rudder_OrVec)
    tail_bac.RequestWingOrientation2InertiaFrame (tail_OrVec)
    
    LU_bac.RequestWingPlaneDirection()
    LD_bac.RequestWingPlaneDirection()
    RU_bac.RequestWingPlaneDirection()
    RD_bac.RequestWingPlaneDirection()
    rudder_bac.RequestWingPlaneDirection()
    tail_bac.RequestWingPlaneDirection()
    
    LU_velVec = LU_wing.getVelocity()
    LD_velVec = LD_wing.getVelocity()
    RU_velVec = RU_wing.getVelocity()
    RD_velVec = RD_wing.getVelocity()
    rudder_velVec = rudder.getVelocity()
    tail_velVec = tail.getVelocity()
    
    
    TheFlapper_velVec = TheFlapper.getVelocity()
    # This function returns a vector containing exactly 6 values. 
    # The first three are respectively the linear velocities in the x, y and z direction. 
    # The last three are respectively the angular velocities around the x, y and z axes
    Flapper_Rotation_current_T = np.transpose(Flapper_Rotation_current)
    Flapper_Angular_velocity_current = np.matmul(Flapper_Rotation_current_T,\
         np.matrix([[TheFlapper_velVec[3]],[TheFlapper_velVec[4]],[TheFlapper_velVec[5]]]))
    # TheFlapper_velVec[0:3]
    # LU_RD_velVec = LU_RD_joint.getVelocity()
    # print('LU_velVec:',LU_velVec)
    # print('tail_velVec:',tail_velVec)
    
    Last_Real_motor_LU_wing_joint_sensor_value = Real_motor_LU_wing_joint_sensor_value
    Last_Real_motor_LD_wing_joint_sensor_value = Real_motor_LD_wing_joint_sensor_value
    Last_Real_motor_RU_wing_joint_sensor_value = Real_motor_RU_wing_joint_sensor_value
    Last_Real_motor_RD_wing_joint_sensor_value = Real_motor_RD_wing_joint_sensor_value
    Last_Real_motor_rudder_joint_sensor_value = Real_motor_rudder_joint_sensor_value
    Last_Real_motor_H_tail_joint_sensor_value = Real_motor_H_tail_joint_sensor_value
    
    Real_motor_LU_wing_joint_sensor_value = Real_motor_LU_wing_joint_sensor.getValue()
    Real_motor_LD_wing_joint_sensor_value = Real_motor_LD_wing_joint_sensor.getValue()
    Real_motor_RU_wing_joint_sensor_value = Real_motor_RU_wing_joint_sensor.getValue()
    Real_motor_RD_wing_joint_sensor_value = Real_motor_RD_wing_joint_sensor.getValue()
    Real_motor_H_tail_joint_sensor_value  = Real_motor_rudder_joint_sensor.getValue()
    
    d_Real_motor_LU_wing_joint_sensor_value = (Real_motor_LU_wing_joint_sensor_value - Last_Real_motor_LU_wing_joint_sensor_value) /Simulation_Gap
    d_Real_motor_LD_wing_joint_sensor_value = (Real_motor_LD_wing_joint_sensor_value - Last_Real_motor_LD_wing_joint_sensor_value) /Simulation_Gap
    d_Real_motor_RU_wing_joint_sensor_value = (Real_motor_RU_wing_joint_sensor_value - Last_Real_motor_RU_wing_joint_sensor_value) /Simulation_Gap
    d_Real_motor_RD_wing_joint_sensor_value = (Real_motor_RD_wing_joint_sensor_value - Last_Real_motor_RD_wing_joint_sensor_value) /Simulation_Gap
    d_Real_motor_rudder_joint_sensor_value  = (Real_motor_rudder_joint_sensor_value - Last_Real_motor_rudder_joint_sensor_value) /Simulation_Gap
    d_Real_motor_H_tail_joint_sensor_value  = (Real_motor_H_tail_joint_sensor_value - Last_Real_motor_rudder_joint_sensor_value) /Simulation_Gap
    # print('d_Real_motor_RU_wing_joint_sensor_value',d_Real_motor_RU_wing_joint_sensor_value)
    
    
    LU_bac.RequestVelocities(vel_FreeFlow,LU_velVec[0:3],\
                                          LU_velVec[3:6],d_Real_motor_LU_wing_joint_sensor_value)
    LD_bac.RequestVelocities(vel_FreeFlow,LD_velVec[0:3],\
                                          LD_velVec[3:6],d_Real_motor_LD_wing_joint_sensor_value)
    RU_bac.RequestVelocities(vel_FreeFlow,RU_velVec[0:3],\
                                          RU_velVec[3:6],d_Real_motor_RU_wing_joint_sensor_value)
    RD_bac.RequestVelocities(vel_FreeFlow,RD_velVec[0:3],\
                                          RD_velVec[3:6],d_Real_motor_RD_wing_joint_sensor_value)
    Period_length = int( round( 1 / StrokeFreq / Simulation_Gap ) )
    # print ('Period_length', Period_length)
    induced_velocity_record_length = induced_velocity_record.__len__()
    if ( Period_length < induced_velocity_record_length ):
        # print('Period_length', type(Period_length))
        # print('induced_velocity_record_length', type(induced_velocity_record_length))
        ave_of_vel_in_Period_length = \
            sum ( induced_velocity_record[ - Period_length : induced_velocity_record_length] ) \
            / Period_length
        # print('ave_of_vel_in_Period_length',ave_of_vel_in_Period_length)
        caudal_direction  = [0,0,-1]
        caudal_direction_in_inertia_raw = np. matmul( Flapper_Rotation_current, caudal_direction)
        caudal_direction_in_inertia     = np.array(caudal_direction_in_inertia_raw).flatten().tolist()
        # print('caudal_direction_in_inertia', caudal_direction_in_inertia)
        mag_in_caudal_ave_vel       = sum (np. multiply(ave_of_vel_in_Period_length, \
                                           caudal_direction_in_inertia ) )
        # mag_in_caudal_ave_vel           = mag_in_caudal_ave_vel_raw[0, 0]
        # print('mag_in_caudal_ave_vel', mag_in_caudal_ave_vel)
        if (mag_in_caudal_ave_vel < 0):
            mag_in_caudal_ave_vel = 0 
        
        trace_back_length = round( FromWing2Tail / (mag_in_caudal_ave_vel + FORCE_RELATIVECONSTANT) /   Simulation_Gap)
        if (trace_back_length < induced_velocity_record_length):
            Flapping_wing_induced_flow_Tail = induced_velocity_record[ - trace_back_length]
        else:
            Flapping_wing_induced_flow_Tail = 0
    while (induced_velocity_record_list_length < induced_velocity_record.__len__()):
        induced_velocity_record.pop(0)
    
    # print('Flapping_wing_induced_flow_Tail:',Flapping_wing_induced_flow_Tail)
    # print('vel_FreeFlow:',vel_FreeFlow)
    # print('rudder_velVec[0:3]:',rudder_velVec[0:3])
    # print('tail_velVec[0:3]:',tail_velVec[0:3])

    rudder_bac.RequestVelocities(vel_FreeFlow + 2 * Flapping_wing_induced_flow_Tail, rudder_velVec[0:3],\
                                          rudder_velVec[3:6],d_Real_motor_rudder_joint_sensor_value)
    # DEBUGtail_FreeFlow                       = np.array([0,0,0])
    #set as 0 0 0 when everything ok
    # print('vel_FreeFlow',vel_FreeFlow)
    tail_bac.RequestVelocities(vel_FreeFlow + 2 *Flapping_wing_induced_flow_Tail, tail_velVec[0:3],\
                                          tail_velVec[3:6],0)                                     
    # TheFlapper_velVec = TheFlapper.getVelocity()
    # print('Flapping_wing_induced_flow:',Flapping_wing_induced_flow)
    # print('itself_moving_flow:',rudder_velVec[0:3])
    # print('TheFlapper_velVec[0:3]:',TheFlapper_velVec[0:3])
    
                                        
    LU_bac.CalcEffectiveVelocity()
    LD_bac.CalcEffectiveVelocity()
    RU_bac.CalcEffectiveVelocity()
    RD_bac.CalcEffectiveVelocity()
    rudder_bac.CalcEffectiveVelocity()
    # print('Tail___________')
    tail_bac.CalcEffectiveVelocity()
    
    LU_bac.CalcAoA()
    LD_bac.CalcAoA()
    RU_bac.CalcAoA()
    RD_bac.CalcAoA()
    rudder_bac.CalcAoA()
    tail_bac.CalcAoA()
    
    LU_bac.CopmputeAerodynamicForce()
    LD_bac.CopmputeAerodynamicForce()
    RU_bac.CopmputeAerodynamicForce()
    RD_bac.CopmputeAerodynamicForce()
    rudder_bac.CopmputeAerodynamicForce()
    tail_bac.CopmputeAerodynamicForce()

    # print('LU_r_shift:',LU_r_shift)
    # LU_r_shift = np.matmul(LU_wing_Rotation_matrix, np.array([LU_bac. X_pos_r, -LU_bac. Y_pos_r, 0]))
    LU_r_shift_in_wing = np.array([LU_bac. X_pos_r, LU_bac. Y_pos_r, 0])
    LU_r_shift = np.array(np.matmul(LU_wing_Rotation_matrix, LU_r_shift_in_wing)).squeeze().tolist()
    LU_r = np.array(np.sum(LU_bac. F_r, axis=0)).squeeze()
    # LU_r_Axis = P2AA(LU_r).tolist()
    LU_r_position = (np.array(LU_wing.getPosition()) + np.array(LU_r_shift)).tolist()
    # LU_r_norm     = np.linalg.norm(LU_r)
    # LU_r_FAM.update_force_device(LU_r_Axis,LU_r_position,LU_r_norm)
    
    LU_a_shift_in_wing = np.array([LU_bac. X_pos_a, LU_bac. Y_pos_a, 0])
    LU_a_shift = np.array(np.matmul(LU_wing_Rotation_matrix, LU_a_shift_in_wing)).squeeze().tolist()
    LU_a = np.array(np.sum(LU_bac. F_a, axis=0)).squeeze()
    # LU_a_Axis = P2AA(LU_a).tolist()
    LU_a_position = (np.array(LU_wing.getPosition()) + np.array(LU_a_shift)).tolist()
    # LU_a_norm     = np.linalg.norm(LU_a)
    # LU_a_FAM.update_force_device(LU_a_Axis,LU_a_position,LU_a_norm)

    LU_t_shift_in_wing = np.array([LU_bac. X_pos_t, LU_bac. Y_pos_t, 0])
    LU_t_shift = np.array(np.matmul(LU_wing_Rotation_matrix, LU_t_shift_in_wing)).squeeze().tolist()
    LU_drag = np.array(np.sum(LU_bac. F_t_drag, axis=0)).squeeze()
    LU_drag_position = (np.array(LU_wing.getPosition()) + np.array(LU_t_shift)).tolist()



    LU_lift = np.array(np.sum(LU_bac. F_t_lift, axis=0)).squeeze()
    # LU_lift_Axis = P2AA(LU_lift).tolist()
    LU_lift_position = LU_drag_position
    # LU_lift_norm     = np.linalg.norm(LU_lift)
    # LU_lift_FAM.update_force_device(LU_lift_Axis,LU_lift_position,LU_lift_norm)
    
    # print('LU_lift_norm:',LU_lift_norm)
    
    LD_drag = np.array(np.sum(LD_bac. F_t_drag, axis=0)).squeeze()
    # LD_drag_Axis = P2AA(LD_drag).tolist()
    LD_drag_position = LD_wing.getPosition()
    # LD_drag_norm     = np.linalg.norm(LD_drag)
    # LD_drag_FAM.update_force_device(LD_drag_Axis,LD_drag_position,LD_drag_norm)
    
    LD_lift = np.array(np.sum(LD_bac. F_t_lift, axis=0)).squeeze()
    # LD_lift_Axis = P2AA(LD_lift).tolist()
    LD_lift_position = LD_drag_position
    # LD_lift_norm     = np.linalg.norm(LD_lift)
    # LD_lift_FAM.update_force_device(LD_lift_Axis,LD_lift_position,LD_lift_norm)
    
    RU_lift = np.array(np.sum(RU_bac. F_t_lift, axis=0)).squeeze()
    RD_lift = np.array(np.sum(RD_bac. F_t_lift, axis=0)).squeeze()
    
    RU_drag = np.array(np.sum(RU_bac. F_t_drag, axis=0)).squeeze()
    RD_drag = np.array(np.sum(RD_bac. F_t_drag, axis=0)).squeeze()

    
    LD_r = np.array(np.sum(LD_bac. F_r, axis=0)).squeeze()
    RU_r = np.array(np.sum(RU_bac. F_r, axis=0)).squeeze()
    RD_r = np.array(np.sum(RD_bac. F_r, axis=0)).squeeze()
    
    # print('LD_r_norm:',np.linalg.norm(LD_r))
    # print('RU_r_norm:',np.linalg.norm(RU_r))
    # print('RD_r_norm:',np.linalg.norm(RD_r))
    # LD_r_Axis = P2AA(LD_r).tolist()
    
    LD_a = np.array(np.sum(LD_bac. F_a, axis=0)).squeeze()
    RU_a = np.array(np.sum(RU_bac. F_a, axis=0)).squeeze()
    RD_a = np.array(np.sum(RD_bac. F_a, axis=0)).squeeze()


    
    rudder_lift = np.array(np.sum(rudder_bac. F_t_lift, axis=0)).squeeze()
    rudder_drag = np.array(np.sum(rudder_bac. F_t_drag, axis=0)).squeeze()
    rudder_r = np.array(np.sum(rudder_bac. F_r, axis=0)).squeeze()
    rudder_a = np.array(np.sum(rudder_bac. F_a, axis=0)).squeeze()
    

    rudder_t_shift_in_local = np.array([rudder_bac. X_pos_t, rudder_bac. Y_pos_t, 0]).squeeze()
    rudder_t_shift = np.array(np.matmul(rudder_Rotation_matrix, rudder_t_shift_in_local)).squeeze().tolist()
  


    tail_lift = np.array(np.sum(tail_bac. F_t_lift, axis=0)).squeeze()
    tail_drag = np.array(np.sum(tail_bac. F_t_drag, axis=0)).squeeze()
    tail_r = np.array(np.sum(tail_bac. F_r, axis=0)).squeeze()
    tail_a = np.array(np.sum(tail_bac. F_a, axis=0)).squeeze()
    
    
    # print('tail_lift',tail_lift)
    # print('tail_drag',tail_drag)  
    JUSTFOTAILRDEBUG = 1
    
    tail_t_shift_in_local = np.array([tail_bac. X_pos_t, tail_bac. Y_pos_t, 0]).squeeze()
    # print('tail_t_shift_in_local:',tail_t_shift_in_local)
    
    tail_t_shift_raw      = np.array(tail_t_shift_in_local).squeeze()
    tail_t_shift = np.array(np.matmul(tail_Rotation_matrix, tail_t_shift_raw )).squeeze().tolist()
    
    tail_drag_position = (np.array(tail.getPosition()) + np.array(tail_t_shift)).tolist()
  
    
    # rudder_lift_position = tail_drag_position
    
    LU_wing.addForceWithOffset([LU_drag[0],LU_drag[1],LU_drag[2]],[LU_bac.X_pos_t,LU_bac.Y_pos_t,0],False)
    LD_wing.addForceWithOffset([LD_drag[0],LD_drag[1],LD_drag[2]],[LD_bac.X_pos_t,LD_bac.Y_pos_t,0],False)
    RU_wing.addForceWithOffset([RU_drag[0],RU_drag[1],RU_drag[2]],[RU_bac.X_pos_t,RU_bac.Y_pos_t,0],False)
    RD_wing.addForceWithOffset([RD_drag[0],RD_drag[1],RD_drag[2]],[RD_bac.X_pos_t,RD_bac.Y_pos_t,0],False)
    rudder .addForceWithOffset([rudder_drag[0],rudder_drag[1],rudder_drag[2]],rudder_t_shift_in_local.tolist(),False)
    tail   .addForceWithOffset([tail_drag[0],tail_drag[1],tail_drag[2]] ,(tail_t_shift_raw ) .tolist() ,False)

    
    LU_wing.addForceWithOffset([LU_lift[0],LU_lift[1],LU_lift[2]],[LU_bac.X_pos_t,LU_bac.Y_pos_t,0],False)
    LD_wing.addForceWithOffset([LD_lift[0],LD_lift[1],LD_lift[2]],[LD_bac.X_pos_t,LD_bac.Y_pos_t,0],False)
    RU_wing.addForceWithOffset([RU_lift[0],RU_lift[1],RU_lift[2]],[RU_bac.X_pos_t,RU_bac.Y_pos_t,0],False)
    RD_wing.addForceWithOffset([RD_lift[0],RD_lift[1],RD_lift[2]],[RD_bac.X_pos_t,RD_bac.Y_pos_t,0],False)
    rudder .addForceWithOffset([rudder_lift[0],rudder_lift[1],rudder_lift[2]],rudder_t_shift_in_local.tolist(),False)
    tail   .addForceWithOffset([tail_lift[0],tail_lift[1],tail_lift[2]] ,(tail_t_shift_raw ) .tolist(),False)
    
    
    LU_lift_in_real = np.array(np.matmul(Flapper_Rotation_current_T, LU_lift)).squeeze()
    LD_lift_in_real = np.array(np.matmul(Flapper_Rotation_current_T, LD_lift)).squeeze()
    RU_lift_in_real = np.array(np.matmul(Flapper_Rotation_current_T, RU_lift)).squeeze()
    RD_lift_in_real = np.array(np.matmul(Flapper_Rotation_current_T, RD_lift)).squeeze()
    
    Total_lift_in_real =  (LU_lift_in_real + LD_lift_in_real + RU_lift_in_real + RD_lift_in_real)
    
    # print('TOTAL_LIFT', [Total_lift_in_real[0],Total_lift_in_real[1],Total_lift_in_real[2]])
    
    Flapping_wing_induced_flow_raw_list = - 0.5 * GetUnitDirection_Safe(Total_lift_in_real) * \
                                np.sqrt( 0.5 * np.linalg.norm(Total_lift_in_real) /\
                                         S_d_Flapping_wing_actuator_disk_area / arena_Air_Density )
    
    Flapping_wing_induced_flow_raw_mat  = np.matrix([[Flapping_wing_induced_flow_raw_list[0]],\
                                                     [Flapping_wing_induced_flow_raw_list[1]],\
                                                     [Flapping_wing_induced_flow_raw_list[2]]])
    Flapping_wing_induced_flow = 2 * np.array(np.matmul(Flapper_Rotation_current, Flapping_wing_induced_flow_raw_mat)).squeeze()
    # print('Flapping_wing_induced_flow', Flapping_wing_induced_flow)
    
    induced_velocity_record.append( Flapping_wing_induced_flow )
    
    LU_wing.addForceWithOffset([LU_r[0],LU_r[1],LU_r[2]],[LU_bac.X_pos_r,LU_bac.Y_pos_r,0],False)
    LD_wing.addForceWithOffset([LD_r[0],LD_r[1],LD_r[2]],[LD_bac.X_pos_r,LD_bac.Y_pos_r,0],False)
    RU_wing.addForceWithOffset([RU_r[0],RU_r[1],RU_r[2]],[RU_bac.X_pos_r,RU_bac.Y_pos_r,0],False)
    RD_wing.addForceWithOffset([RD_r[0],RD_r[1],RD_r[2]],[RD_bac.X_pos_r,RD_bac.Y_pos_r,0],False)
    
  

    LU_wing.addForceWithOffset([LU_a[0],LU_a[1],LU_a[2]],[LU_bac.X_pos_a,LU_bac.Y_pos_a,0],False)
    LD_wing.addForceWithOffset([LD_a[0],LD_a[1],LD_a[2]],[LD_bac.X_pos_a,LD_bac.Y_pos_a,0],False)
    RU_wing.addForceWithOffset([RU_a[0],RU_a[1],RU_a[2]],[RU_bac.X_pos_a,RU_bac.Y_pos_a,0],False)
    RD_wing.addForceWithOffset([RD_a[0],RD_a[1],RD_a[2]],[RD_bac.X_pos_a,RD_bac.Y_pos_a,0],False)
    
   
   
   ####-------------Control Tasks-------------------
    # print('d_Real_motor_LU_wing_joint_sensor_value',d_Real_motor_LU_wing_joint_sensor_value)
    
    torsion_spring_constant = 0.025
    # torsion_spring_yaw_offset = 0 ##0-0.3
    
    # K_pitch = 5
    # torsion_spring_pitch_offset =  K_pitch * (Flapper_OrVec_in_Euler[1]-0.6)### 仔细调整
    
    
    K_mo = 0.15
    
     
    
    #first order filter
    # Filtered_anular_velocity = Filtered_anular_velocity + 0.01 * (Flapper_Angular_velocity_current - Filtered_anular_velocity)
    
    #second order filter update
    Angular_velocity_filter. march_forward(Flapper_Angular_velocity_current)
    
    #FT observer update
    Flapper_translation_value = Flapper_translation.getSFVec3f()
    Flapper_pos             = np.mat( [[Flapper_translation_value[0]],\
                                       [Flapper_translation_value[1]],\
                                       [Flapper_translation_value[2]]]) 
    
    
    u_t_in_body_fixed_frame = np.mat( [[0],[0],[0.2]])  
    u_t_in_inertia_frame = Flapper_Rotation_current * u_t_in_body_fixed_frame 
    
    Here_pos_observer. march_forward(u_t_in_inertia_frame, Flapper_pos)
    
    Gamma_now = Flapper_Rotation_current_T * np.mat( [0,0,1] ).T
      
    
    e_13 = np.mat([[1],[0],[1]])

    Pi = Flapper_Rotation_current * e_13

    Flapper_psi = np.arctan2( Pi[1,0],Pi[0,0])
    
    Flapper_psi_filter. march_forward(Flapper_psi)
    mat_TheFlapper_velVec = np.matrix([[TheFlapper_velVec[0]],[TheFlapper_velVec[1]],[TheFlapper_velVec[2]]])

    
    if ( np.mod(RecordCount, Controller_Gap_vs_Simulation_Gap)==0):
    # These values relate to the FWAV mounting.
        torsion_spring_pitch_offset = 0
        torsion_spring_yaw_offset  = 0       
        
        print('RecordCount:',RecordCount)
        track_time = RecordCount / 1000.0
        # print('Flapper_psi:', Flapper_psi)
        
        Gamma_des_x = 0
        Gamma_des_y = 0
        if T_reader.get_x_pos(track_time) is not None:
            p_d = np.mat([[T_reader.get_x_pos(track_time)],\
                            [T_reader.get_y_pos(track_time)], \
                            [T_reader.get_z_pos(track_time)]])
            print('p_d:',p_d)
            print('Flapper_pos:',Flapper_pos)
            v_d = np.mat([[T_reader.get_x_vel(track_time)],\
                            [T_reader.get_y_vel(track_time)], \
                            [T_reader.get_z_vel(track_time)]])
            print('v_d:',v_d)
            Postion_Controller.Calc_u_t(p_d, Flapper_pos,\
                                    v_d, mat_TheFlapper_velVec,\
                                    Flapper_psi_filter.Get_filtered(),\
                                    Flapper_psi_filter.Get_filtered_D())
   
            GammaP =  np.mat([Postion_Controller.Gamma_xp,##This is not validated yet
                            Postion_Controller.Gamma_yp,
                            Postion_Controller.Gamma_zp]).T
        else:
            GammaP = np.mat([0, 0 ,1]).T # This line is used for testing the pure reduced attitude controller
            v_d = np.mat([  [0.0], [0.0], [0.0]])
            Postion_Controller.Calc_u_t(p_d, Flapper_pos,\
                                    v_d, mat_TheFlapper_velVec,\
                                    Flapper_psi_filter.Get_filtered(),\
                                    Flapper_psi_filter.Get_filtered_D())
            
        GammaP = 1 / np.linalg.norm(GammaP) * GammaP
        
        # GammaP = np.mat([-1, 0 ,0.5]).T
        # GammaP = 1 / np.linalg.norm(GammaP) * GammaP
        print('GammaP:',GammaP)
        Gamma = Flapper_Rotation_current.T * np.mat([0,0,1]).T
        
        Attitude_Controller.Calc_u(GammaP, Gamma, Flapper_Angular_velocity_current)

        K_pitch = 0.3
        K_roll = 0.3
        
        pitch_com =  K_pitch * Attitude_Controller.theta_ele
        roll_com  =  K_roll  * Attitude_Controller.theta_rud
        
     
        
        alt_now = Flapper_translation_value[2]
        alt_vel_des = 0
        # alt_vel_des = -0.3
        
        alt_des = p_d[2,0]
        
        alt_vel = v_d[2,0]
        
        alt_last = alt_now
        
        k_i = 0.1
        k_p = 10
        k_d = 3
        
        e_alt = k_p * (alt_des - alt_now)
        e_alt_vel = k_d * (alt_vel_des - alt_vel)
        
        e_alt = max (- alt_err_mag_max, min(alt_err_mag_max, e_alt))
        # e_alt_vel = max (- alt_vel_mag_max, min(alt_vel_mag_max, e_alt_vel))
      
        
        alt_int = alt_int + k_i * (alt_des - alt_now)
        alt_int = max (- alt_int_mag_max, min(alt_int_mag_max, alt_int))
        
        
        
        StrokeFreq_def =  10.5
        StrokeFreq = StrokeFreq_def  + e_alt + e_alt_vel
        
        # StrokeFreq = np.sqrt(Postion_Controller.f_flap_2)
        
        StrokeFreq_min = 7
        StrokeFreq_max = 15
        
        StrokeFreq = max( StrokeFreq_min,\
                                min ( StrokeFreq, StrokeFreq_max))
        
        print('StrokeFreq:',StrokeFreq)
        # d_omega = np.mat( [d_omega_x, d_omega_y, 0] ).T 
        # tau_now = Inertia_matrix @ d_omega \
        #         - np.cross( omega_now.T,  (Inertia_matrix @ omega_now).T).T
        
        print('roll_com:',roll_com)
        print('pitch_com:',pitch_com)
                  
        roll_rudder_amplitude = roll_com
        H_tail_amplitude =   pitch_com
        
        # print('Gamma_des:',Gamma_des)
        # print('Gamma_now:',Gamma_now)
        
        
        roll_rudder_amplitude = max( - roll_rudder_mag_max, \
                                min ( roll_rudder_amplitude, roll_rudder_mag_max))
        
        H_tail_amplitude = max ( - H_tail_mag_max, \
                                min (H_tail_amplitude, H_tail_mag_max))
        
        
        
        # print('torsion_spring_yaw_offset',torsion_spring_yaw_offset)
    
    
    # print('Flapper position', Flapper_translation_value)
    
    # K_height = 5
    
        
    # print('Height:', Flapper_translation_value[1])
    # print('StrokeFreq',StrokeFreq)
    # StrokeFreq = 15
            
    # print('Flapper_OrVec_in_Euler[1],', Flapper_OrVec_in_Euler[1])    
    # print('TEST: torsion_spring_pitch_offset,', torsion_spring_pitch_offset)
    Real_motor_H_tail_joint. setPosition(H_tail_amplitude)
    
    Real_motor_rudder_joint. setPosition(roll_rudder_amplitude)
    
    Real_motor_LU_wing_joint.setTorque(- torsion_spring_constant \
                                       * (Real_motor_LU_wing_joint_sensor_value - torsion_spring_yaw_offset - torsion_spring_pitch_offset))
    Real_motor_LD_wing_joint.setTorque(- torsion_spring_constant \
                                       * (Real_motor_LD_wing_joint_sensor_value - torsion_spring_yaw_offset - torsion_spring_pitch_offset))
    Real_motor_RU_wing_joint.setTorque(- torsion_spring_constant \
                                       * (Real_motor_RU_wing_joint_sensor_value - torsion_spring_yaw_offset + torsion_spring_pitch_offset))
    Real_motor_RD_wing_joint.setTorque(- torsion_spring_constant \
                                       * (Real_motor_RD_wing_joint_sensor_value - torsion_spring_yaw_offset + torsion_spring_pitch_offset))
    
    # print('StrokeFreq:',StrokeFreq)
    
    LD_t_shift_in_wing = np.array([LD_bac. X_pos_t, LD_bac. Y_pos_t, 0])
    LD_t_shift = np.array(np.matmul(LD_wing_Rotation_matrix, LD_t_shift_in_wing)).squeeze().tolist()
    LD_t_position = (np.array(LD_wing.getPosition()) + np.array(LD_t_shift)).tolist()
    
    LD_r_shift_in_wing = np.array([LD_bac. X_pos_r, LD_bac. Y_pos_r, 0])
    LD_r_shift = np.array(np.matmul(LD_wing_Rotation_matrix, LD_r_shift_in_wing)).squeeze().tolist()
    LD_r_position = (np.array(LD_wing.getPosition()) + np.array(LD_r_shift)).tolist()
    
    LD_a_shift_in_wing = np.array([LD_bac. X_pos_a, LD_bac. Y_pos_a, 0])
    LD_a_shift = np.array(np.matmul(LD_wing_Rotation_matrix, LD_a_shift_in_wing)).squeeze().tolist()
    LD_a_position = (np.array(LD_wing.getPosition()) + np.array(LD_a_shift)).tolist()
    
    
    
    RU_t_shift_in_wing = np.array([RU_bac. X_pos_t, RU_bac. Y_pos_t, 0])
    RU_t_shift = np.array(np.matmul(RU_wing_Rotation_matrix, RU_t_shift_in_wing)).squeeze().tolist()
    RU_t_position = (np.array(RU_wing.getPosition()) + np.array(RU_t_shift)).tolist()
    
    RU_r_shift_in_wing = np.array([RU_bac. X_pos_r, RU_bac. Y_pos_r, 0])
    RU_r_shift = np.array(np.matmul(RU_wing_Rotation_matrix, RU_r_shift_in_wing)).squeeze().tolist()
    RU_r_position = (np.array(RU_wing.getPosition()) + np.array(RU_r_shift)).tolist()
    
    RU_a_shift_in_wing = np.array([RU_bac. X_pos_a, RU_bac. Y_pos_a, 0])
    RU_a_shift = np.array(np.matmul(RU_wing_Rotation_matrix, RU_a_shift_in_wing)).squeeze().tolist()
    RU_a_position = (np.array(RU_wing.getPosition()) + np.array(RU_a_shift)).tolist()
    
    
    
    RD_t_shift_in_wing = np.array([RD_bac. X_pos_t, RD_bac. Y_pos_t, 0])
    RD_t_shift = np.array(np.matmul(RD_wing_Rotation_matrix, RD_t_shift_in_wing)).squeeze().tolist()
    RD_t_position = (np.array(RD_wing.getPosition()) + np.array(RD_t_shift)).tolist()
    
    RD_r_shift_in_wing = np.array([RD_bac. X_pos_r, RD_bac. Y_pos_r, 0])
    RD_r_shift = np.array(np.matmul(RD_wing_Rotation_matrix, RD_r_shift_in_wing)).squeeze().tolist()
    RD_r_position = (np.array(RD_wing.getPosition()) + np.array(RD_r_shift)).tolist()
    
    RD_a_shift_in_wing = np.array([RD_bac. X_pos_a, RD_bac. Y_pos_a, 0])
    RD_a_shift = np.array(np.matmul(RD_wing_Rotation_matrix, RD_a_shift_in_wing)).squeeze().tolist()
    RD_a_position = (np.array(RD_wing.getPosition()) + np.array(RD_a_shift)).tolist()
    
    #### torque
    rudder_drag_position = (np.array(rudder.getPosition()) + np.array(rudder_t_shift)).tolist()
    rudder_lift_position = rudder_drag_position

    
    LU_lift_torque = CTR (LU_lift, np.array(Flapper_translation_value) - np.array(LU_lift_position))
    LU_drag_torque = CTR (LU_drag, np.array(Flapper_translation_value) - np.array(LU_drag_position))
    LU_r_torque    = CTR (LU_r,    np.array(Flapper_translation_value) - np.array(LU_r_position))
    LU_a_torque    = CTR (LU_a,    np.array(Flapper_translation_value) - np.array(LU_a_position))
    
    LD_lift_torque = CTR (LD_lift, np.array(Flapper_translation_value) - np.array(LD_t_position))
    LD_drag_torque = CTR (LD_drag, np.array(Flapper_translation_value) - np.array(LD_t_position))
    LD_r_torque    = CTR (LD_r,    np.array(Flapper_translation_value) - np.array(LD_r_position))
    LD_a_torque    = CTR (LD_a,    np.array(Flapper_translation_value) - np.array(LD_a_position))
    
    RU_lift_torque = CTR (RU_lift, np.array(Flapper_translation_value) - np.array(RU_t_position))
    RU_drag_torque = CTR (RU_drag, np.array(Flapper_translation_value) - np.array(RU_t_position))
    RU_r_torque    = CTR (RU_r,    np.array(Flapper_translation_value) - np.array(RU_r_position))
    RU_a_torque    = CTR (RU_a,    np.array(Flapper_translation_value) - np.array(RU_a_position))
    
    RD_lift_torque = CTR (RD_lift, np.array(Flapper_translation_value) - np.array(RD_t_position))
    RD_drag_torque = CTR (RD_drag, np.array(Flapper_translation_value) - np.array(RD_t_position))
    RD_r_torque    = CTR (RD_r,    np.array(Flapper_translation_value) - np.array(RD_r_position))
    RD_a_torque    = CTR (RD_a,    np.array(Flapper_translation_value) - np.array(RD_a_position))
    
    
    Total_wing_torque = LU_lift_torque + LU_drag_torque + LU_r_torque + LU_a_torque +\
                        LD_lift_torque + LD_drag_torque + LD_r_torque + LD_a_torque +\
                        RU_lift_torque + RU_drag_torque + RU_r_torque + RU_a_torque +\
                        RD_lift_torque + RD_drag_torque + RD_r_torque + RD_a_torque
    
    Total_wing_force  = LU_lift+ LU_drag + LU_r + LU_a +\
                        LD_lift + LD_drag + LD_r + LD_a +\
                        RU_lift + RU_drag + RU_r + RU_a +\
                        RD_lift + RD_drag + RD_r + RD_a
                        
    # print('Total_Torque:', Total_wing_torque)
    
    rudder_lift_torque  = CTR (rudder_lift, np.array(Flapper_translation_value) - np.array(rudder_lift_position))
    rudder_drag_torque  = CTR (rudder_drag, np.array(Flapper_translation_value) - np.array(rudder_drag_position))
    
    Total_rudder_torque = rudder_lift_torque + rudder_drag_torque   
    Total_rudder_force  = rudder_lift + rudder_drag
    
    
    tail_lift_torque    = CTR (tail_lift, np.array(Flapper_translation_value) - np.array(tail_drag_position))
    tail_drag_torque    = CTR (tail_drag, np.array(Flapper_translation_value) - np.array(tail_drag_position))
    Total_tail_torque   = tail_lift_torque + tail_lift_torque
    
    Total_tail_force    = tail_lift + tail_drag
    
    RecordCount = RecordCount + 1
    
    if (RecordCount < mat_length):
        Total_body_rotation_list.    append(Flapper_Rotation_current)
        Total_body_translation_list. append(Flapper_translation_value)
        # Total_body_translation_desire. append(p_d)
        
        Total_body_angular_velocity_list. append(Flapper_Angular_velocity_current)
        Total_Angular_velocity_filtered. append(Angular_velocity_filter. Get_filtered())

        Total_desire_gamma_list. append(GammaP)
        Total_current_gamma_list. append(Gamma_now)
        
        Total_current_alt_vel_des. append(alt_vel)
        Total_current_alt_vel. append(alt_vel_des)
        
        # Total_yaw_input. append(torsion_spring_yaw_offset)
        Total_roll_input. append(roll_rudder_amplitude)
        Total_pitch_input. append(H_tail_amplitude)
        Total_Freq_stroke. append(StrokeFreq)
        
        Total_record_v_list.  append(mat_TheFlapper_velVec)
        Total_p_d_list.  append(p_d)
        Total_v_d_list. append(v_d)
        
    if (RecordCount == mat_length):
    
        # ,Gamma_des_y_i,alt_des_vel_i)
       
        scio.savemat(File_name +  mat_file_format, { 'record_Flapper_att':Total_body_rotation_list,\
                                  'record_p':Total_body_translation_list,\
                                  'record_v':Total_record_v_list,\
                                  'record_Angle_vel':Total_body_angular_velocity_list,\
                                  'Total_Angular_velocity_filtered_list':Total_Angular_velocity_filtered,\
                                  'record_Gammap':Total_desire_gamma_list,\
                                  'record_Gamma':Total_current_gamma_list,\
                                  'Total_current_alt_vel_des':Total_current_alt_vel_des,\
                                  'Total_current_alt_vel':Total_current_alt_vel,\
                                  'Total_roll_input':Total_roll_input,\
                                  'Total_pitch_input':Total_pitch_input,\
                                  'Total_Freq_stroke':Total_Freq_stroke,\
                                  'record_p_d': Total_p_d_list,\
                                  'record_v_d': Total_v_d_list}) 

        
        Total_body_rotation_list.    clear()
        Total_body_translation_list. clear()
        Total_body_angular_velocity_list. clear()
        Total_Angular_velocity_filtered. clear()
        
        Total_desire_gamma_list.clear()
        Total_current_gamma_list.clear()
        
        Total_current_alt_vel_des.clear()
        Total_current_alt_vel.clear()
        
        Total_roll_input. clear()
        Total_pitch_input. clear()
        
        Total_Freq_stroke. clear()
        Total_record_v_list. clear()
        Total_p_d_list.clear()
        Total_v_d_list.clear()
        
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("#%Y_%m_%d_%H_%M_%S#")
        File_name = 'Newdata/' + formatted_datetime + Category
        RecordCount = 0
        
        ### Set initial postion
        initial_translation = [random.uniform(-init_dist, init_dist), \
                                random.uniform(-init_dist, init_dist), \
                                random.uniform(-init_dist, init_dist)]
        
        flapper.simulationReset()
        # Flapper_translation.setSFVec3f(initial_translation)
        # Flapper_rotation.setSFRotation(initial_rotation_in_axis_angle)
        
        
        Gamma_now = np.mat([0,0,1]).T
        Gamma_last = Gamma_now
        alt_int = 0.0
        alt_now = 0.0
        alt_last = 0.0
        alt_des = 0.0

        # RecordCount = 0
        # episode_i = episode_i + 1
        
        # Gamma_des_x_i = np.mod(episode_i, Gamma_des_x_list_num)
        # Gamma_des_y_i = np.mod( int(episode_i / Gamma_des_x_list_num),  Gamma_des_y_list_num)
        # alt_des_vel_i = int(episode_i / Gamma_des_x_list_num / Gamma_des_y_list_num)
        
        
        
        # if Gamma_des_x_i >= Gamma_des_x_list_num:
        #     Gamma_des_x_i = 0
        #     Gamma_des_y_i = Gamma_des_y_i + 1
        
        # if Gamma_des_y_i >= Gamma_des_y_list_num:
        #     Gamma_des_y_i = 0
        #     alt_des_vel_i = alt_des_vel_i + 1
    
      
    
    
    
    