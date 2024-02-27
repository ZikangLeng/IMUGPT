import sys
sys.path.append('/mnt/d/Research/GaTech/code/video2imu_v2/util/transfer')
from ProbabilityIntegralTransfer import prob_integ_xfer
import numpy as np
from pprint import pprint
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

dir_root = '/mnt/d/Research/GaTech'
dir_data = dir_root + '/data/subtle_activity'


file_real = '/mnt/d/Research/GaTech/data/subtle_activity/real_all.npz'
Xy_real = np.load(file_real)
X_real = Xy_real['X']
y_real = Xy_real['y']
print ('load from ...', file_real, X_real.shape, y_real.shape)

file_vir_raw = '/mnt/d/Research/GaTech/data/subtle_activity/vir_raw_all.npz'
Xy_vir_raw = np.load(file_vir_raw)
X_vir_raw = Xy_vir_raw['X']
y_vir_raw = Xy_vir_raw['y']
print('load from ...', file_vir_raw, X_vir_raw.shape, y_vir_raw.shape)

dict_label = {'violin': 0, 
            'Driving_Automatic': 1, 
            'Reading': 2, 
            'Cutting_Components': 3, 
            'Writting_on_Paper': 4, 
            'Eat_with_hand': 5, 
            'Wiping': 6, 
            'Washes_dishes': 7, 
            'Sweeping': 8, 
            'Driving_Manual': 9, 
            'Playing_on_Guitar': 10, 
            'Washing_Hands': 11, 
            'Piano': 12, 
            'Drawing': 13, 
            'Flipping': 14, 
            'shower': 15, 
            'Bed_making': 16}

perClass = True

X_vir_cal = prob_integ_xfer(X_vir_raw, X_real,
                    y_vir=y_vir_raw, y_real=y_real, 
                    perClass=perClass, dict_class_sel=dict_label)
y_vir_cal = y_vir_raw
# print (X_vir_cal.shape)

file_save = dir_data + '/vir_cal_all_hk.npz'
np.savez(file_save, X=X_vir_cal, y=y_vir_cal)
print ('save in ...', file_save, X_vir_cal.shape, y_vir_cal.shape)
assert False

''' 
Visualize distribution 
'''
dir_save = '/mnt/d/Research/GaTech/exps/subtle_activity/calib_hk'

n_sample = 1000


colors = ['r', 'g', 'b']

if 1:
  ''' per class & per channel plot & per data x-axis
  '''
  dir_save_class = dir_save + '/class'
  os.makedirs(dir_save_class, exist_ok=True)

  ys = np.unique(y_real)

  for y in ys:
    ''' real '''
    idx_c = y_real == y
    X_real_c = X_real[idx_c]

    ''' vir raw '''
    idx_c = y_vir_raw == y
    X_vir_raw_c = X_vir_raw[idx_c]

    ''' vir cal '''
    idx_c = y_vir_cal == y
    X_vir_cal_c = X_vir_cal[idx_c]

    for ch in range(3):
      fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,5))


      ''' real '''
      min_x, max_x = np.amin(X_real_c[:,ch]), np.amax(X_real_c[:,ch])
      x_plot = np.linspace(min_x, max_x, n_sample)
      log_dens, bin_edges = np.histogram(X_real_c[:,ch], bins=x_plot, density=True)
      axes[0].plot(x_plot[:-1], log_dens, color=colors[ch], label='real_X')
      axes[0].legend()

      ''' vir raw '''
      min_x, max_x = np.amin(X_vir_raw_c[:,ch]), np.amax(X_vir_raw_c[:,ch])
      x_plot = np.linspace(min_x, max_x, n_sample)
      log_dens, bin_edges = np.histogram(X_vir_raw_c[:,ch], bins=x_plot, density=True)
      axes[1].plot(x_plot[:-1], log_dens, color=colors[ch], label='vir_raw_X')
      axes[1].legend()

      '''vir cal '''
      min_x, max_x = np.amin(X_vir_cal_c[:,ch]), np.amax(X_vir_cal_c[:,ch])
      x_plot = np.linspace(min_x, max_x, n_sample)
      log_dens, bin_edges = np.histogram(X_vir_cal_c[:,ch], bins=x_plot, density=True)
      axes[2].plot(x_plot[:-1], log_dens, color=colors[ch], label='vir_cal_X')
      axes[2].legend()

      file_fig = dir_save_class + f'/dist_calib_x-axis_{y}_{ch}.png'
      plt.savefig(file_fig)
      print ('save in ...', file_fig)

if 1:
  ''' per channel plot & per data x-axis
  '''

  for ch in range(3):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,5))


    ''' real '''
    min_x, max_x = np.amin(X_real[:,ch]), np.amax(X_real[:,ch])
    x_plot = np.linspace(min_x, max_x, n_sample)
    log_dens, bin_edges = np.histogram(X_real[:,ch], bins=x_plot, density=True)
    axes[0].plot(x_plot[:-1], log_dens, color=colors[ch], label='real_X')
    axes[0].legend()

    ''' vir raw '''
    min_x, max_x = np.amin(X_vir_raw[:,ch]), np.amax(X_vir_raw[:,ch])
    x_plot = np.linspace(min_x, max_x, n_sample)
    log_dens, bin_edges = np.histogram(X_vir_raw[:,ch], bins=x_plot, density=True)
    axes[1].plot(x_plot[:-1], log_dens, color=colors[ch], label='vir_raw_X')
    axes[1].legend()

    '''vir cal '''
    min_x, max_x = np.amin(X_vir_cal[:,ch]), np.amax(X_vir_cal[:,ch])
    x_plot = np.linspace(min_x, max_x, n_sample)
    log_dens, bin_edges = np.histogram(X_vir_cal[:,ch], bins=x_plot, density=True)
    axes[2].plot(x_plot[:-1], log_dens, color=colors[ch], label='vir_cal_X')
    axes[2].legend()

    file_fig = dir_save + f'/dist_calib_x-axis_{ch}.png'
    plt.savefig(file_fig)
    print ('save in ...', file_fig)