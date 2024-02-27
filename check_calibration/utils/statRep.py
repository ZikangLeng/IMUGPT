import numpy as np
from scipy import stats


def statRep(data):
    M_T = np.mean(data,axis=0)
    V_T = np.var(data,axis=0)
    SK_T = stats.skew(data,axis=0)
    K_T = stats.kurtosis(data,axis=0)
    RMS_T = np.sqrt(np.mean(data**2,axis=0))

    H_T = np.hstack((M_T,V_T))
    H_T = np.hstack((H_T,SK_T))
    H_T = np.hstack((H_T,K_T))
    H_T = np.hstack((H_T,RMS_T))
    return H_T