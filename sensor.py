import os
import numpy as np
import subprocess
import shlex
import socket

import sys

from imusim.platforms.imus import IdealIMU, Orient3IMU
from imusim.simulation.base import Simulation
from imusim.behaviours.imu import BasicIMUBehaviour
from imusim.io.bvh import BVHLoader
from imusim.trajectories.rigid_body import SplinedBodyModel
from imusim.environment.base import Environment
from imusim.simulation.calibrators import ScaleAndOffsetCalibrator

import argparse
import pickle as cp
import numpy as np

hostname = socket.gethostname()

import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

# change for different skeletons 
# closest_joint_from_sensor = { 'Left_hip': 'Left_hip', 
#  'Left_knee': 'Left_knee', 
#  'Left_ankle': 'Left_ankle',
#  'Right_hip': 'Right_hip', 
#  'Right_knee': 'Right_knee',
#   'Right_ankle': 'Right_ankle',
#   'Spine1': 'Spine1',
#   'Spine2': 'Spine2',
#   'Spine3': 'Spine3',
#   'Head': 'Head',
#   'Left_shoulder': 'Left_shoulder',
#   'Left_elbow': 'Left_elbow',
#   'Left_wrist': 'Left_wrist', 
#   'Right_shoulder': 'Right_shoulder',
#   'Right_elbow': 'Right_elbow', 
#   'Right_wrist': 'Right_wrist'}

# #realworld 
# closest_joint_from_sensor = { 'Left_knee': 'Left_knee',
#   'Left_ankle': 'Left_ankle',
#    'Head': 'Head',
#    'Left_wrist': 'Left_wrist', 
#     'Spine1': 'Spine1',
#     'Spine3': 'Spine3',
#    'Left_elbow': 'Left_elbow'}

#pamap 

# double check
version = 'ideal' 
# version = 'sim'

_samplingPeriod = 0.
calibSamples = 1000
calibRotVel = 20

def extract_vir_imu(file_bvh, file_acc, file_gyro, dataset):

  if dataset == 'realworld':
    closest_joint_from_sensor = { 'Left_knee': 'Left_knee',
  'Left_ankle': 'Left_ankle',
   'Head': 'Head',
   'Left_wrist': 'Left_wrist', 
    'Spine1': 'Spine1',
    'Spine3': 'Spine3',
   'Left_elbow': 'Left_elbow'}
  elif dataset == 'pamap':
    closest_joint_from_sensor = {
  'Right_ankle': 'Right_ankle',
   'Right_wrist': 'Right_wrist', 
    'Spine3': 'Spine3'}
  elif dataset == 'usc-had':
    closest_joint_from_sensor = {
  'Right_hip': 'Right_hip',
   'Right_knee': 'Right_knee', }
  else:
    raise ValueError(f"{dataset} is not valid ")
  
  print(closest_joint_from_sensor)


  list_joint2sensor = {}
  for sensor_name in closest_joint_from_sensor:
    joint_name = closest_joint_from_sensor[sensor_name]
    list_joint2sensor[joint_name] = sensor_name    


  sensor = {}  

  # if os.path.exists(file_acc) and os.path.exists(file_gyro):
  #   acc = np.load(file_acc)
  #   gyro = np.load(file_gyro)

  #   sensor = {
  #     'acc': acc,
  #     'gyro': gyro}
  #   return sensor

  # Extact virtual sensor with imusim
  # updated to python 3 version
  path_bvh = file_bvh
  path_acc = file_acc
  path_gyro = file_gyro

  # load mocap
  with open(path_bvh, 'r') as bvhFile:
    conversionFactor = 1
    loader = BVHLoader(bvhFile,conversionFactor)
    loader._readHeader()
    loader._readMotionData()
    model = loader.model
  print ('load mocap from ...', path_bvh)
  #print (model)

  # spline intrepolation
  splinedModel = SplinedBodyModel(model)
  startTime = splinedModel.startTime
  endTime = splinedModel.endTime

  if _samplingPeriod == 0.:
    samplingPeriod = (endTime - startTime)/loader.frameCount
  else:
    samplingPeriod = _samplingPeriod
  print ('frameCount:', loader.frameCount)
  print ('samplingPeriod:', samplingPeriod)

  if version == 'ideal':
    print ('Simulating ideal IMU.')

    # set simulation
    sim = Simulation()
    sim.time = startTime

    # run simulation
    dict_imu = {}
    for joint_name in list_joint2sensor:
      imu = IdealIMU()
      imu.simulation = sim
      imu.trajectory = splinedModel.getJoint(joint_name)

      BasicIMUBehaviour(imu, samplingPeriod)

      dict_imu[joint_name] = imu

    sim.run(endTime)

  elif version == 'sim':
    print ('Simulating Orient3IMU.')
    
    # set simulation
    env = Environment()
    calibrator = ScaleAndOffsetCalibrator(env, calibSamples, samplingPeriod, calibRotVel)
    sim = Simulation(environment=env)
    sim.time = startTime

    # run simulation
    dict_imu = {}
    for joint_name in list_joint2sensor:

      imu = Orient3IMU()
      calibration = calibrator.calibrate(imu)
      print ('imu calibration:', joint_name)
      
      imu.simulation = sim
      imu.trajectory = splinedModel.getJoint(joint_name)

      BasicIMUBehaviour(imu, samplingPeriod, calibration, initialTime=sim.time)

      dict_imu[joint_name] = imu

    sim.run(endTime)

  # collect sensor values
  acc_seq = {}
  gyro_seq = {}
  for joint_name in list_joint2sensor:
    sensor_name = list_joint2sensor[joint_name]
    imu = dict_imu[joint_name]

    if version == 'ideal':    
      acc_seq[sensor_name] = imu.accelerometer.rawMeasurements.values.T
      gyro_seq[sensor_name] = imu.gyroscope.rawMeasurements.values.T
    elif version == 'sim':
      acc_seq[sensor_name] = imu.accelerometer.calibratedMeasurements.values.T
      gyro_seq[sensor_name] = imu.gyroscope.calibratedMeasurements.values.T
    
    print (sensor_name, acc_seq[sensor_name].shape, gyro_seq[sensor_name].shape)
  
  # save
  np.savez(path_acc, **acc_seq)
  print ('save in ...', path_acc)
  np.savez(path_gyro, **gyro_seq)
  print ('save in ...', path_gyro)		

  #-----------------------------

  # acc = np.load(file_acc)
  # gyro = np.load(file_gyro)
  acc = acc_seq
  gyro = gyro_seq

  sensor = {
    'acc': acc,
    'gyro': gyro}

  # if cfg.sensor.is_render:
  #   for pID in sensor:
  #     dir_sensor_pID = cfg.sensor.dir_result + '/pID_{}'.format(pID)

  #     acc = sensor[pID]['acc']
  #     gyro = sensor[pID]['gyro']
  #     ts = pID_info[pID]['ts']

  #     for joint in acc.files:
  #       # print(acc[joint].shape)
  #       # print(gyro[joint].shape)
        
  #       len_seq = np.amin((acc[joint].shape[0], 
  #                         gyro[joint].shape[0], 
  #                         ts.shape[0]))
  #       acc_joint = acc[joint][:len_seq]
  #       gyro_joint = gyro[joint][:len_seq]
  #       ts = ts[:len_seq]    

  #       fig = plt.figure(figsize=(10,5))
  #       fig.suptitle('{} | {}'.format(pID, joint))

  #       ax = fig.add_subplot(6,1,1)
  #       ax.set_title('Accelerometer')
  #       ax.plot(ts, acc_joint[:,0], 'r')
  #       ax.set_ylabel('x')
  #       ax = fig.add_subplot(6,1,2)
  #       ax.plot(ts, acc_joint[:,1], 'g')
  #       ax.set_ylabel('y')
  #       ax = fig.add_subplot(6,1,3)
  #       ax.plot(ts, acc_joint[:,2], 'b')
  #       ax.set_ylabel('z')

  #       ax = fig.add_subplot(6,1,4)
  #       ax.set_title('Gyroscope')
  #       ax.plot(ts, gyro_joint[:,0], 'r')
  #       ax.set_ylabel('x')
  #       ax = fig.add_subplot(6,1,5)
  #       ax.plot(ts, gyro_joint[:,1], 'g')
  #       ax.set_ylabel('y')
  #       ax = fig.add_subplot(6,1,6)
  #       ax.plot(ts, gyro_joint[:,2], 'b')
  #       ax.set_ylabel('z')

  #       file_savefig = dir_sensor_pID + '/{}.png'.format(joint)
  #       fig.subplots_adjust(hspace=.5)
  #       plt.savefig(file_savefig)
  #       print ('save in ...', file_savefig)
  
  return sensor