import multiprocessing as mp
import my_funcs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

experiments = ['equil_1e-0.5','equil_1e-1.0','equil_1e-1.6','equil_1e-1.8','equil_1e-2.0','equil_1e-2.2','snr1','snr10','snr25','snr30','snr35','snr100','sparse10','sparse30','sparse40','sparse50','sparse80']
num_processors = 8

def all_func(snr):
    exp_path = './Experiments/{}/'.format(snr)
    num_trials = 50

    my_funcs.get_exp_design(snr,exp_path,num_trials)

    best_id = my_funcs.get_best_trial(snr,exp_path,num_trials)

    my_funcs.get_pics_and_pde_params(best_id,snr,exp_path)

if __name__ == '__main__':
        p=mp.Pool(processes = num_processors)
        try:
            p.map(all_func,experiments)
        except Exception as e:
            print(f'Failed with: {e}')
        p.close()
        p.join()