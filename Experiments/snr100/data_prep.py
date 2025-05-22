import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

def prep_training_data():

    ########################
    ### Training Dataset ###
    ########################

    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    ### Import Training Dataset ###
    cell1, neigh1, fs1 = np.load('sim_data/training/training1.npz')['cell'], np.load('sim_data/training/training1.npz')['neigh'], np.load('sim_data/training/training1.npz')['future']
    cell2, neigh2, fs2 = np.load('sim_data/training/training2.npz')['cell'], np.load('sim_data/training/training2.npz')['neigh'], np.load('sim_data/training/training2.npz')['future']
    cell3, neigh3, fs3 = np.load('sim_data/training/training3.npz')['cell'], np.load('sim_data/training/training3.npz')['neigh'], np.load('sim_data/training/training3.npz')['future']
    cell4, neigh4, fs4 = np.load('sim_data/training/training4.npz')['cell'], np.load('sim_data/training/training4.npz')['neigh'], np.load('sim_data/training/training4.npz')['future']
    cell5, neigh5, fs5 = np.load('sim_data/training/training5.npz')['cell'], np.load('sim_data/training/training5.npz')['neigh'], np.load('sim_data/training/training5.npz')['future']
    cell6, neigh6, fs6 = np.load('sim_data/training/training6.npz')['cell'], np.load('sim_data/training/training6.npz')['neigh'], np.load('sim_data/training/training6.npz')['future']
    cell7, neigh7, fs7 = np.load('sim_data/training/training7.npz')['cell'], np.load('sim_data/training/training7.npz')['neigh'], np.load('sim_data/training/training7.npz')['future']
    cell8, neigh8, fs8 = np.load('sim_data/training/training8.npz')['cell'], np.load('sim_data/training/training8.npz')['neigh'], np.load('sim_data/training/training8.npz')['future']
    cell9, neigh9, fs9 = np.load('sim_data/training/training9.npz')['cell'], np.load('sim_data/training/training9.npz')['neigh'], np.load('sim_data/training/training9.npz')['future']
    cell10, neigh10, fs10 = np.load('sim_data/training/training10.npz')['cell'], np.load('sim_data/training/training10.npz')['neigh'], np.load('sim_data/training/training10.npz')['future']

    # restore np.load for future normal usage
    np.load = np_load_old

    cell = np.concatenate((cell1,cell2,cell3,cell4,cell5,cell6,cell7,cell8,cell9,cell10))
    neigh = np.concatenate((neigh1,neigh2,neigh3,neigh4,neigh5,neigh6,neigh7,neigh8,neigh9,neigh10))
    fs = np.concatenate((fs1,fs2,fs3,fs4,fs5,fs6,fs7,fs8,fs9,fs10))

    x = cell
    x_neigh = neigh
    x_dot = fs-cell
    x_plus = fs

    for i in range(x_dot.shape[1]):
        np.save('sim_data/training/x_dot_abs_{}_max.npy'.format(i),np.abs(x_dot[:,i]).max())
        x_dot[:,i] = x_dot[:,i]/np.abs(x_dot[:,i]).max()

    n_samples = int(x.shape[0]*1.00) # Sample data 60%
    sample_indices = np.random.randint(0,x.shape[0],n_samples)
    x = np.take(x,sample_indices,axis=0)
    x_neigh = np.take(x_neigh,sample_indices,axis=0)
    x_plus = np.take(x_plus,sample_indices,axis=0)
    x_dot = np.take(x_dot,sample_indices,axis=0)

    np.save('sim_data/training/x.npy',x)
    np.save('sim_data/training/x_neigh.npy',x_neigh)
    np.save('sim_data/training/x_plus.npy',x_plus)
    np.save('sim_data/training/x_dot.npy',x_dot)

