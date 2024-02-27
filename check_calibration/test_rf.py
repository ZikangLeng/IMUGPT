import os, sys
import utils
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report


'''
RF with ECDF
Real Only =>
[0.4847584316571028, 0.5306919841084805, 0.4602276580051588, 0.419090209559865, 0.49177367502708796]
Avg: 0.47730839167153893
portion 0.05 real
portion 0.1 real
[0.3662179746384647, 0.3299265865954538, 0.25284816940349647, 0.16585436335347942, 0.22573197739067172]                      
Avg: 0.26811581427631326
portion 0.2 real
[0.44395132981496466, 0.3808735655510859, 0.2762188062731096, 0.17683526726962845, 0.257479085997664]
Avg: 0.30707161098129054
portion 0.3 real
[0.4855995770001069, 0.45169016436581866, 0.29362273229891384, 0.20489983910659573, 0.3193682194103088]
Avg: 0.35103610643634875
portion 0.4 real
[0.4924759462142968, 0.4648779022404062, 0.33113777209138623, 0.23496866779210776, 0.3487986351476592]
Avg: 0.3744517846971712
portion 0.5 real
[0.49277079021000897, 0.47297388321754286, 0.3670418685113046, 0.2573559056997989, 0.3865458384880568]
Avg: 0.3953376572253425
--------------
Vir Only =>
[0.17952196648963176, 0.20210746314170458, 0.2003827189280312, 0.19304934923204628, 0.23915930956795992]
Avg: 0.20284416147187473
--------------
Real + Vir
[0.4624096159949643, 0.5170585593610659, 0.4614585246764463, 0.3908653068746737, 0.48569088607442495]
Avg: 0.4634965785963151
portion 0.05 real
[0.283235800780688, 0.28449200565036603, 0.22180236195902064, 0.18029530727282572, 0.2660356013045974]
0.24717221539349957
portion 0.1 real
[0.34089209994493846, 0.31378490906708817, 0.25191737420356514, 0.20730342297102833, 0.28517323333942785]
Avg: 0.2798142079052096
portion 0.2 real
[0.3989803970301778, 0.3858099914918743, 0.2960180201229228, 0.21760899796864921, 0.3028918179043848]
Avg: 0.3202618449036018
portion 0.3 real
[0.45041306893029, 0.43206776469549457, 0.3161241193608505, 0.2249838384705744, 0.34589567606743715]
Avg: 0.3538968935049293
portion 0.4 real
[0.4573224319708262, 0.4506669395408231, 0.3412427049347179, 0.23533105711645988, 0.35741639351961874]
Avg: 0.3683959054164892
portion 0.5 real
[0.4609032632799915, 0.453919799580175, 0.37352763217399626, 0.25024511419588213, 0.3864306021540558]
0.3850052822768201
'''

dir_root = '/mnt/d/Research/GaTech'
dir_data = 'D:\CBAResearch\IMUTube-percom\data'
dir_save = 'D:\CBAResearch\IMUTube-percom\save'

file_real = dir_data +'/real_all.npz'
Xy_real = np.load(file_real)
X_real = Xy_real['X']
y_real = Xy_real['y']
print ('load from ...', file_real, X_real.shape, y_real.shape)

file_vir = dir_data + '/vir_cal_all_hk.npz'
Xy_vir = np.load(file_vir)
X_vir = Xy_vir['X']
y_vir = Xy_vir['y']
print ('load from ...', file_vir, X_vir.shape, y_vir.shape)

'''
Sliding Window
'''

frame_size_seconds = 3
step_size_seconds = int(frame_size_seconds/2)
sampling_rate = 25

frame_size = frame_size_seconds * sampling_rate
step_size = step_size_seconds * sampling_rate

''' real '''
X_real_win = utils.sliding_window(X_real,
      (frame_size, X_real.shape[1]),
      (step_size, 1))

y_real_win = np.asarray([
  [i[-1]] for i in utils.sliding_window(
    y_real, 
    frame_size, 
    step_size)])
y_real_win = y_real_win.reshape(len(y_real_win)).astype(np.uint8)

''' vir '''
X_vir_win = utils.sliding_window(X_vir,
      (frame_size, X_vir.shape[1]),
      (step_size, 1))

y_vir_win = np.asarray([
  [i[-1]] for i in utils.sliding_window(
    y_vir, 
    frame_size, 
    step_size)])
y_vir_win = y_vir_win.reshape(len(y_vir_win)).astype(np.uint8)

file_save = dir_save + '/real_win.npz'
np.savez(file_save, X=X_real_win, y=y_real_win)
print ('save in ...', file_save, X_real_win.shape, y_real_win.shape)

file_save = dir_save + '/vir_win.npz'
np.savez(file_save, X=X_vir_win, y=y_vir_win)
print ('save in ...', file_save, X_vir_win.shape, y_vir_win.shape)

'''
Feature extraction
'''

ecdf_num = 10

''' real '''
_feat = utils.ecdfRep(X_real_win[0], components=ecdf_num)
#_feat = utils.statRep(X_real_win[0])
X_real_feat = np.empty((X_real_win.shape[0], _feat.shape[0]))
for i in range(X_real_win.shape[0]):
  X_real_feat[i] = utils.ecdfRep(X_real_win[i], components=ecdf_num)
  #X_real_feat[i] = utils.statRep(X_real_win[i])

''' vir '''
_feat = utils.ecdfRep(X_vir_win[0], components=ecdf_num)
#_feat = utils.statRep(X_vir_win[0])
X_vir_feat = np.empty((X_vir_win.shape[0], _feat.shape[0]))
for i in range(X_vir_win.shape[0]):
  X_vir_feat[i] = utils.ecdfRep(X_vir_win[i], components=ecdf_num)
  #X_vir_feat[i] = utils.statRep(X_vir_win[i])

file_save = dir_save + '/real_ecdf.npz'
np.savez(file_save, X=X_real_feat, y=y_real_win)
print ('save in ...', file_save, X_real_feat.shape, y_real_win.shape)

file_save = dir_save + '/vir_ecdf.npz'
np.savez(file_save, X=X_vir_feat, y=y_vir_win)
print ('save in ...', file_save, X_vir_feat.shape, y_vir_win.shape)

'''
Experiment
'''

# portion_val = np.arange(11)/10
# portion_val[0] = 0.05
# result = np.zeros(11)

#for i in range(len(portion_val)):
portion = 1

#exp_protocol = 'real' # real / vir / real+vir
#exp_protocol = 'vir' # real / vir / real+vir
exp_protocol = 'real+vir' # real / vir / real+vir

list_f1_score = []
f1_per_cls = np.zeros(17)
num_fold = 5
skf = StratifiedKFold(n_splits=5)
for train_idx, test_idx in skf.split(X_real_feat, y_real_win):
  X_real_train, y_real_train = X_real_feat[train_idx], y_real_win[train_idx]
  X_real_test, y_real_test = X_real_feat[test_idx], y_real_win[test_idx]

  clf = RandomForestClassifier(n_estimators=185)

  ''' portion real '''
  if portion == 1.:
    X_real_por = X_real_train
    y_real_por = y_real_train
  else:
    ys = np.unique(y_real_train)

    X_real_por = np.empty(X_real_train.shape)
    y_real_por = np.empty(y_real_train.shape)

    i_start = 0
    for y in ys:
      idx_y = y_real_train == y
      num_por = int(np.sum(idx_y)*portion)

      X_real_y = X_real_train[idx_y]
      y_real_y = y_real_train[idx_y]

      X_real_por[i_start:i_start+num_por] = X_real_y[:num_por]
      y_real_por[i_start:i_start+num_por] = y_real_y[:num_por]

      i_start += num_por

    X_real_por = X_real_por[:i_start]
    y_real_por = y_real_por[:i_start]

  print (f'portion {portion}:', X_real_por.shape, y_real_por.shape)
  
  print (exp_protocol)
  if exp_protocol == 'real':
    ''' train only with real '''
    X_train = X_real_por
    y_train = y_real_por
    # print (X_real_por.shape, y_real_por.shape)
  elif exp_protocol == 'vir':
    ''' train only with vir '''
    X_train = X_vir_feat
    y_train = y_vir_win
    # print (X_vir_feat.shape, y_vir_win.shape)
  elif exp_protocol == 'real+vir':
    X_train = np.concatenate([X_real_por, X_vir_feat])
    y_train = np.concatenate([y_real_por, y_vir_win])
    # print (X_real_por.shape, y_real_por.shape)
    # print (X_vir_feat.shape, y_vir_win.shape)
    # print (X_train.shape, y_train.shape)
  else:
    assert False, exp_protocol

  ''' standardize '''
  X_mean = np.mean(X_train, axis=0).reshape((1,-1))
  X_std = np.mean(X_train, axis=0).reshape((1,-1))
  X_train -= X_mean
  X_train /= X_std
  X_real_test -= X_mean
  X_real_test /= X_std

  clf.fit(X_train,y_train)

  pred_test = clf.predict(X_real_test)

  f1 = f1_score(y_real_test, pred_test, average='macro')
  report = classification_report(y_real_test, pred_test, output_dict= True)
  print(f1)

  for i in range(np.unique(y_real_train).shape[0]):
    f1_per_cls[i] += report[f'{i}']['f1-score']

  list_f1_score.append(f1)


f1_per_cls /= num_fold
print (list_f1_score)
print (np.mean(list_f1_score))
print(f1_per_cls)


file_save = dir_save + "/f1_per_cls_mix_1.txt"
np.savetxt(file_save, f1_per_cls)


# result[i] = np.mean(list_f1_score)

# print(result)