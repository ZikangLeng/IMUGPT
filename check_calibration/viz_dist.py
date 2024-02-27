import os
import pickle as cp
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

dir_root = '/mnt/d/Research/GaTech'
dir_data = dir_root + '/data/subtle_activity'

dir_save = '/mnt/d/Research/GaTech/exps/subtle_activity'

'''
Load Dataset
'''

''' Load Real '''
file_real = dir_data + '/unsmoothed_real.pkl'
sess_X_real, sess_y_real = cp.load(open(file_real, 'rb')) 
print ('load from ...', file_real, len(sess_X_real), len(sess_y_real))

X_real, y_real = [], []
for X, y in zip(sess_X_real, sess_y_real):
  assert np.all(np.isfinite(X)), [X.shape, np.where(np.any(np.isfinite(X), axis=1))]
  # print (X.shape)
  # print (y)

  y = np.ones((X.shape[0],)) * y
  X_real.append(X)
  y_real.append(y)
  # print ('----')

  # assert False

X_real = np.concatenate(X_real)
y_real = np.concatenate(y_real)
assert np.all(np.isfinite(X_real))
assert np.all(np.isfinite(y_real))

file_save = dir_data + '/real_all.npz'
np.savez(file_save, X=X_real, y=y_real)
print ('save in ...',  file_save, X_real.shape, y_real.shape)

''' Load Vir Raw '''
file_vir_raw = dir_data + '/uncalibrated_vir.pkl'
sess_X_vir_raw, sess_y_vir_raw = cp.load(open(file_vir_raw, 'rb'))
print ('load from ...', file_vir_raw, len(sess_X_vir_raw), len(sess_y_vir_raw))

X_vir_raw, y_vir_raw = [], []
for X, y in zip(sess_X_vir_raw, sess_y_vir_raw):
  # assert np.all(np.isfinite(X)), [X.shape, np.where(np.any(np.isfinite(X), axis=1))]
  if np.all(np.isfinite(X)):
    y = np.ones((X.shape[0],)) * y
    X_vir_raw.append(X)
    y_vir_raw.append(y)

X_vir_raw = np.concatenate(X_vir_raw)
y_vir_raw = np.concatenate(y_vir_raw)
assert np.all(np.isfinite(X_vir_raw)), [np.where(np.any(np.isfinite(X_vir_raw), axis=1))]
assert np.all(np.isfinite(y_vir_raw))

file_save = dir_data + '/vir_raw_all.npz'
np.savez(file_save, X=X_vir_raw, y=y_vir_raw)
print ('save in ...', file_save, X_vir_raw.shape, y_vir_raw.shape)
# assert False

''' Load Vir Cal '''
file_vir_cal = dir_data + '/calibrated_vir.pkl'
sess_X_vir_cal, sess_y_vir_cal = cp.load(open(file_vir_cal, 'rb'))
print ('load from ...', file_vir_cal, len(sess_X_vir_cal), len(sess_y_vir_cal))

X_vir_cal, y_vir_cal = [], []
for X, y in zip(sess_X_vir_cal, sess_y_vir_cal):
  assert np.all(np.isfinite(X)), [X.shape, np.where(np.any(np.isfinite(X), axis=1))]
  y = np.ones((X.shape[0],)) * y
  X_vir_cal.append(X)
  y_vir_cal.append(y)

X_vir_cal = np.concatenate(X_vir_cal)
y_vir_cal = np.concatenate(y_vir_cal)
assert np.all(np.isfinite(X_vir_cal))
assert np.all(np.isfinite(y_vir_cal))

file_save = dir_data + '/vir_cal_all.npz'
np.savez(file_save, X=X_vir_cal, y=y_vir_cal)
print ('save in ...', file_save, X_vir_cal.shape, y_vir_cal.shape)
assert False

'''
Check distribution
'''
if 0:
  q = np.arange(0, 101, 5)
  # print (q)
  # assert False

  channels = ['X', 'Y', 'Z']

  for ch in range(3):
    print (f'real {channels[ch]}:')
    pcts_real = np.percentile(X_real[:,ch], q)
    print (pcts_real)

    print (f'vir raw {channels[ch]}:')
    pcts_vir_raw = np.percentile(X_vir_raw[:,ch], q)
    print (pcts_vir_raw)

    print (f'vir cal {channels[ch]}:')
    pcts_vir_cal = np.percentile(X_vir_cal[:,ch], q)
    print (pcts_vir_cal)
    print ('-----------------')

  assert False

''' 
Visualize distribution 
'''

n_sample = 1000


colors = ['r', 'g', 'b']

if 0:
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

if 0:
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

if 0:
  ''' per channel plot & shared x-axis
  '''

  for ch in range(3):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,5))

    min_x = np.amin([
      np.amin(X_real[:,ch]),
      np.amin(X_vir_raw[:,ch]),
      np.amin(X_vir_cal[:,ch])])

    max_x = np.amax([
      np.amax(X_real[:,ch]),
      np.amax(X_vir_raw[:,ch]),
      np.amax(X_vir_cal[:,ch])])

    x_plot = np.linspace(min_x, max_x, n_sample)
    # print (x_plot)
    # assert False

    ''' real '''
    log_dens, bin_edges = np.histogram(X_real[:,ch], bins=x_plot, density=True)
    axes[0].plot(x_plot[:-1], log_dens, color=colors[ch], label='real_X')
    axes[0].legend()

    ''' vir raw '''
    log_dens, bin_edges = np.histogram(X_vir_raw[:,ch], bins=x_plot, density=True)
    axes[1].plot(x_plot[:-1], log_dens, color=colors[ch], label='vir_raw_X')
    axes[1].legend()

    '''vir cal '''
    log_dens, bin_edges = np.histogram(X_vir_cal[:,ch], bins=x_plot, density=True)
    axes[2].plot(x_plot[:-1], log_dens, color=colors[ch], label='vir_cal_X')
    axes[2].legend()

    file_fig = dir_save + f'/dist_calib_{ch}.png'
    plt.savefig(file_fig)
    print ('save in ...', file_fig)
  
if 0:
  ''' shared channel plot & shared x-axis
  '''

  fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,5))
  for ch in range(3):

    min_x = np.amin([
      np.amin(X_real[:,ch]),
      np.amin(X_vir_raw[:,ch]),
      np.amin(X_vir_cal[:,ch])])

    max_x = np.amax([
      np.amax(X_real[:,ch]),
      np.amax(X_vir_raw[:,ch]),
      np.amax(X_vir_cal[:,ch])])

    x_plot = np.linspace(min_x, max_x, n_sample)
    # print (x_plot)
    # assert False

    ''' real '''
    log_dens, bin_edges = np.histogram(X_real[:,ch], bins=x_plot, density=True)
    axes[0, ch].plot(x_plot[:-1], log_dens, color=colors[ch], label='real_X')
    axes[0, ch].legend()

    ''' vir raw '''
    log_dens, bin_edges = np.histogram(X_vir_raw[:,ch], bins=x_plot, density=True)
    axes[1, ch].plot(x_plot[:-1], log_dens, color=colors[ch], label='vir_raw_X')
    axes[1, ch].legend()

    '''vir cal '''
    log_dens, bin_edges = np.histogram(X_vir_cal[:,ch], bins=x_plot, density=True)
    axes[2, ch].plot(x_plot[:-1], log_dens, color=colors[ch], label='vir_cal_X')
    axes[2, ch].legend()

  file_fig = dir_save + f'/dist_calib.png'
  plt.savefig(file_fig)
  print ('save in ...', file_fig)