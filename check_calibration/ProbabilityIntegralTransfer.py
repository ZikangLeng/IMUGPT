import numpy as np

def _prob_integ_xfer(X_vir, X_real, 
                    y_vir=None, y_real=None, 
                    pcts=None,
                    resolution=100, perClass=False, dict_class_sel=None):
  assert X_vir.shape[1] == X_real.shape[1]

  if pcts is None:
    pcts = np.linspace(0, 100, resolution)
  X_vir_trans = np.empty(X_vir.shape)
  X_vir_trans[:] = np.nan

  if perClass:
    assert (y_vir is not None) and (y_real is not None) and (dict_class_sel is not None)
    assert X_vir.shape[0] == y_vir.shape[0]
    assert X_real.shape[0] == y_real.shape[0]

    for i_c, label in enumerate(dict_class_sel):
      idx_act_vir = y_vir == dict_class_sel[label]
      idx_act_real = y_real == dict_class_sel[label]
      X_vir_activity = X_vir[idx_act_vir]
      X_real_activity = X_real[idx_act_real]

      pct_vir = np.percentile(X_vir_activity, pcts, axis=0)
      pct_real = np.percentile(X_real_activity, pcts, axis=0)

      for i_pct in range(len(pcts)):

        for ch in range(X_vir.shape[1]):
          if i_pct == 0:
            idx_range = (X_vir[:,ch] <= pct_vir[i_pct,ch]) \
                        & idx_act_vir
          else:
            idx_range = (X_vir[:,ch] > pct_vir[i_pct-1, ch]) \
                        & (X_vir[:,ch] <= pct_vir[i_pct,ch]) \
                        & idx_act_vir
          
          X_vir_trans[idx_range,ch] = pct_real[i_pct,ch]

          #print (i_c, '{} | pct {} | ch {}'.format(label, i_pct, ch))
        #print (i_c, '{} | pct {}'.format(label, i_pct))
      print (i_c, '{}'.format(label))

  else:
    pct_vir = np.percentile(X_vir, pcts, axis=0)
    pct_real = np.percentile(X_real, pcts, axis=0)
    #print (pcts)
    #print (pct_vir.shape)
    #print (pct_real.shape)

    for i_pct in range(len(pcts)):
      for ch in range(X_vir.shape[1]):
        if i_pct == 0:
          idx_range = X_vir[:,ch] <= pct_vir[i_pct,ch]
        else:
          idx_range = (X_vir[:,ch] > pct_vir[i_pct-1, ch]) & (X_vir[:,ch] <= pct_vir[i_pct,ch])
        
        X_vir_trans[idx_range,ch] = pct_real[i_pct,ch]

        #print ('pct {} | ch {}'.format(i_pct, ch))
        
  assert np.all(np.isfinite(X_vir_trans))

  return X_vir_trans

#----------------------
# multithreading
import threading

def xfer_class_pct_ch(X_vir_trans, X_vir, pct_vir, pct_real, i_pct, ch, idx_act_vir):
  if i_pct == 0:
    idx_range = (X_vir[:,ch] <= pct_vir[i_pct,ch]) \
                & idx_act_vir
  else:
    idx_range = (X_vir[:,ch] > pct_vir[i_pct-1, ch]) \
                & (X_vir[:,ch] <= pct_vir[i_pct,ch]) \
                & idx_act_vir
  
  X_vir_trans[idx_range,ch] = pct_real[i_pct,ch]

def xfer_class(X_vir_trans, i_c, label, 
              X_vir, y_vir, 
              X_real, y_real, 
              pcts, dict_class_sel):
  idx_act_vir = y_vir == dict_class_sel[label]
  idx_act_real = y_real == dict_class_sel[label]
  X_vir_activity = X_vir[idx_act_vir]
  X_real_activity = X_real[idx_act_real]

  pct_vir = np.percentile(X_vir_activity, pcts, axis=0)
  pct_real = np.percentile(X_real_activity, pcts, axis=0)

  list_multithread = []
  for i_pct in range(len(pcts)):
    for ch in range(X_vir.shape[1]):
      list_multithread.append(
        threading.Thread(target=xfer_class_pct_ch,
          args=(X_vir_trans, X_vir, pct_vir, pct_real, i_pct, ch, idx_act_vir)))
      #print (i_c, '{} | pct {} | ch {}'.format(label, i_pct, ch))
    #print (i_c, '{} | pct {}'.format(label, i_pct))

  for mt in list_multithread:
    mt.start()
  
  for mt in list_multithread:
    mt.join()
  
  print (i_c, '{}'.format(label))

def xfer_pct_ch(X_vir_trans, X_vir, pct_vir, pct_real, i_pct, ch):
  if i_pct == 0:
    idx_range = X_vir[:,ch] <= pct_vir[i_pct,ch]
  else:
    idx_range = (X_vir[:,ch] > pct_vir[i_pct-1, ch]) & (X_vir[:,ch] <= pct_vir[i_pct,ch])
  
  X_vir_trans[idx_range,ch] = pct_real[i_pct,ch]

def prob_integ_xfer_mt(X_vir, X_real, 
                    y_vir=None, y_real=None, 
                    pcts=None,
                    resolution=100, perClass=False, dict_class_sel=None):
  assert X_vir.shape[1] == X_real.shape[1]

  if pcts is None:
    pcts = np.linspace(0, 100, resolution)

  X_vir_trans = np.empty(X_vir.shape)
  X_vir_trans[:] = np.nan

  if perClass:
    assert (y_vir is not None) and (y_real is not None) and (dict_class_sel is not None)
    assert X_vir.shape[0] == y_vir.shape[0]
    assert X_real.shape[0] == y_real.shape[0]

    list_multithread = []
    for i_c, label in enumerate(dict_class_sel):
      list_multithread.append(
        threading.Thread(target=xfer_class,
          args=(X_vir_trans, i_c, label, 
            X_vir, y_vir, 
            X_real, y_real, 
            pcts, dict_class_sel)))
    
    for mt in list_multithread:
      mt.start()
    
    for mt in list_multithread:
      mt.join()

  else:
    pct_vir = np.percentile(X_vir, pcts, axis=0)
    pct_real = np.percentile(X_real, pcts, axis=0)

    list_multithread = []
    for i_pct in range(len(pcts)):
      for ch in range(X_vir.shape[1]):
        list_multithread.append(
          threading.Thread(target=xfer_pct_ch, 
            args=(X_vir_trans, X_vir, pct_vir, pct_real, i_pct, ch)))
    
    for mt in list_multithread:
      mt.start()
    
    for mt in list_multithread:
      mt.join()

  assert np.all(np.isfinite(X_vir_trans))

  return X_vir_trans    

# multithreading with queue
# Referred https://www.shanelynn.ie/using-python-threading-for-multiple-results-queue/
# However, nested queue just stucks ... trying concurrent.future.ThreadPoolExecutor
from queue import Queue

def xfer_class_pct_ch_mtq(q2, X_vir_trans, X_vir, 
                          pct_vir, pct_real, 
                          idx_act_vir):

  while not q2.empty():
    work = q2.get()
    i_pct, ch = work

    if i_pct == 0:
      idx_range = (X_vir[:,ch] <= pct_vir[i_pct,ch]) \
                  & idx_act_vir
    else:
      idx_range = (X_vir[:,ch] > pct_vir[i_pct-1, ch]) \
                  & (X_vir[:,ch] <= pct_vir[i_pct,ch]) \
                  & idx_act_vir
    
    X_vir_trans[idx_range,ch] = pct_real[i_pct,ch]

    q2.task_done()

  return True

def xfer_class_mtq(q, X_vir_trans, 
              X_vir, y_vir, 
              X_real, y_real, 
              pcts, dict_class_sel,
              num_threads):
  while not q.empty():
    work = q.get()
    i_c, label = work

    idx_act_vir = y_vir == dict_class_sel[label]
    idx_act_real = y_real == dict_class_sel[label]
    X_vir_activity = X_vir[idx_act_vir]
    X_real_activity = X_real[idx_act_real]

    pct_vir = np.percentile(X_vir_activity, pcts, axis=0)
    pct_real = np.percentile(X_real_activity, pcts, axis=0)

    q2 = Queue(maxsize=0)
    for i_pct in range(len(pcts)):
      for ch in range(X_vir.shape[1]):
        q2.put((i_pct, ch))
    
    for i in range(num_threads):
      worker = threading.Thread(target=xfer_class_pct_ch_mtq,
                args=(q2, X_vir_trans, X_vir, 
                          pct_vir, pct_real, 
                          idx_act_vir))
      # worker.setDaemon(True)
      worker.start()
    
    q2.join()
  
    print (i_c, '{}'.format(label))
    return True

def xfer_pct_ch_mtq(q, X_vir_trans, X_vir, pct_vir, pct_real):
  while not q.empty():
    work = q.get()
    i_pct, ch = work

    if i_pct == 0:
      idx_range = X_vir[:,ch] <= pct_vir[i_pct,ch]
    else:
      idx_range = (X_vir[:,ch] > pct_vir[i_pct-1, ch]) & (X_vir[:,ch] <= pct_vir[i_pct,ch])
    
    X_vir_trans[idx_range,ch] = pct_real[i_pct,ch]

    q.task_done()
  
  return True

def prob_integ_xfer_mtq(X_vir, X_real, 
                    y_vir=None, y_real=None, 
                    pcts=None,
                    resolution=100, perClass=False, dict_class_sel=None,
                    num_threads=50):
  assert X_vir.shape[1] == X_real.shape[1]

  if pcts is None:
    pcts = np.linspace(0, 100, resolution)

  X_vir_trans = np.empty(X_vir.shape)
  X_vir_trans[:] = np.nan

  if perClass:
    raise NotImplementedError('nested queue just stucks ... trying concurrent.future.ThreadPoolExecutor')

    assert (y_vir is not None) and (y_real is not None) and (dict_class_sel is not None)
    assert X_vir.shape[0] == y_vir.shape[0]
    assert X_real.shape[0] == y_real.shape[0]

    q = Queue(maxsize=0)
    for i_c, label in enumerate(dict_class_sel):
      q.put((i_c, label))
    
    for i in range(int(num_threads/10)):
      worker = threading.Thread(target=xfer_class_mtq,
                args=(q, X_vir_trans, 
                  X_vir, y_vir, 
                  X_real, y_real, 
                  pcts, dict_class_sel,
                  10))
      # worker.setDaemon(True)
      worker.start()
    q.join()    

  else:
    pct_vir = np.percentile(X_vir, pcts, axis=0)
    pct_real = np.percentile(X_real, pcts, axis=0)

    q = Queue(maxsize=0)
    for i_pct in range(len(pcts)):
      for ch in range(X_vir.shape[1]):
        q.put((i_pct, ch))
    
    for t in range(num_threads):
      worker = threading.Thread(target=xfer_pct_ch_mtq,
                args=(q, X_vir_trans, X_vir, pct_vir, pct_real))
      worker.setDaemon(True)
      worker.start()
    
    q.join()

  assert np.all(np.isfinite(X_vir_trans))

  return X_vir_trans    

# multithreading with ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

def xfer_class_tpe(X_vir_trans, i_c, label, 
              X_vir, y_vir, 
              X_real, y_real, 
              pcts, dict_class_sel,
              num_threads,
              verbose):
  idx_act_vir = y_vir == dict_class_sel[label]
  idx_act_real = y_real == dict_class_sel[label]
  X_vir_activity = X_vir[idx_act_vir]
  X_real_activity = X_real[idx_act_real]

  pct_vir = np.percentile(X_vir_activity, pcts, axis=0)
  pct_real = np.percentile(X_real_activity, pcts, axis=0)

  with ThreadPoolExecutor(max_workers=num_threads) as executor:
    for i_pct in range(len(pcts)):
      for ch in range(X_vir.shape[1]):
        executor.submit(xfer_class_pct_ch, X_vir_trans, X_vir, pct_vir, pct_real, i_pct, ch, idx_act_vir)

        #print (i_c, '{} | pct {} | ch {}'.format(label, i_pct, ch))
      #print (i_c, '{} | pct {}'.format(label, i_pct))
  if verbose:
    print (i_c, '{}'.format(label))

def prob_integ_xfer(X_vir, X_real, 
                    y_vir=None, y_real=None, 
                    pcts=None,
                    resolution=100, perClass=False, dict_class_sel=None,
                    num_threads=20,
                    verbose=False):
  assert X_vir.shape[1] == X_real.shape[1]

  if pcts is None:
    pcts = np.linspace(0, 100, resolution)

  X_vir_trans = np.empty(X_vir.shape)
  X_vir_trans[:] = np.nan

  if perClass:
    if verbose:
      print ('... supervised probability transfer ...')
    assert (y_vir is not None) and (y_real is not None) and (dict_class_sel is not None)
    assert X_vir.shape[0] == y_vir.shape[0]
    assert X_real.shape[0] == y_real.shape[0]

    with ThreadPoolExecutor(max_workers=int(num_threads/10)) as executor:
      for i_c, label in enumerate(dict_class_sel):
        executor.submit(xfer_class_tpe, 
            X_vir_trans, i_c, label, 
            X_vir, y_vir, 
            X_real, y_real, 
            pcts, dict_class_sel, 10,
            verbose)
  else:
    if verbose:
      print ('... unsupervised probability transfer ...')
    pct_vir = np.percentile(X_vir, pcts, axis=0)
    pct_real = np.percentile(X_real, pcts, axis=0)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
      for i_pct in range(len(pcts)):
        for ch in range(X_vir.shape[1]):
          executor.submit(xfer_pct_ch, 
                X_vir_trans, X_vir, 
                pct_vir, pct_real, i_pct, ch)    

  assert np.all(np.isfinite(X_vir_trans))

  return X_vir_trans    

if __name__ == '__main__':
  '''
  Unit test for thread version
  compare if the output is same for both versions
  '''
  import os, sys
  sys.path.append('/nethome/hkwon64/Research/imuTube/code_video2imu_v2')
  from dataset.freeweights import load_virtual, load_real
  import time

  freq = 30
  clip_pct = 99
  dataset_name = 'MyoGym' # Gym
  target_setting = '13_classes'

  X_vir, y_vir = load_virtual(freq=freq, clip_pct=clip_pct)
  X_real, y_real, s_real, display_labels = load_real(dataset_name, freq=freq, target_setting=target_setting)

  dict_label = {}
  for c, label in enumerate(display_labels):
    dict_label[label] = c

  perClass = True

  if perClass:
    start_time = time.time()
    X_vir_trans = prob_integ_xfer(X_vir, X_real,
                    y_vir=y_vir, y_real=y_real, 
                    perClass=perClass, dict_class_sel=dict_label)
    end_time = time.time()
    print ('X_vir_trans:', X_vir_trans.shape, '| time:', end_time-start_time)

    start_time = time.time()
    X_vir_trans_mt = prob_integ_xfer_mt(X_vir, X_real,
                    y_vir=y_vir, y_real=y_real, 
                    perClass=perClass, dict_class_sel=dict_label)
    end_time = time.time()
    print ('X_vir_trans_mt:', X_vir_trans_mt.shape , '| time:', end_time-start_time)

    assert np.array_equal(X_vir_trans_mt, X_vir_trans)
    print ('With class & MultiThreading Done !')

  else:
    start_time = time.time()
    X_vir_trans = prob_integ_xfer(X_vir, X_real,
                    y_vir=y_vir, y_real=y_real, 
                    perClass=perClass, dict_class_sel=dict_label)
    end_time = time.time()
    print ('X_vir_trans:', X_vir_trans.shape, '| time:', end_time-start_time)
    
    start_time = time.time()
    X_vir_trans_mt = prob_integ_xfer_mt(X_vir, X_real,
                    y_vir=y_vir, y_real=y_real, 
                    perClass=perClass, dict_class_sel=dict_label)
    end_time = time.time()
    print ('X_vir_trans_mt:', X_vir_trans_mt.shape , '| time:', end_time-start_time)

    assert np.array_equal(X_vir_trans_mt, X_vir_trans)
    print ('Without class & MultiThreading Done !')
