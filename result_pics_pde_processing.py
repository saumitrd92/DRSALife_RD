import multiprocessing as mp
import my_funcs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import json

if os.path.exists('config_params.json'):
    config = json.load(open('config_params.json'))

experiments = []

for equil_order in config.get("equil_order_l", [-0.5,-1.0,-1.6,-1.8,-2.0,-2.2]):
        exp_name = 'equil_1e-{}'.format(equil_order)
        experiments.append(exp_name)

for sparsity in config.get("sparsity_l", [10,30,40,50,80]):
    exp_name = 'sparse{}'.format(sparsity)
    experiments.append(exp_name)

for snr in config.get("snr_l", [1,10,25,30,35,100]):
    exp_name = 'snr{}'.format(snr)
    experiments.append(exp_name)

num_processors = config.get("num_processors", 8)

def all_func(exp):
    exp_path = './Experiments/{}/'.format(exp)
    num_trials = config.get("num_trials", 50)

    my_funcs.get_exp_design(exp,exp_path,num_trials)

    best_id = my_funcs.get_best_trial(exp,exp_path,num_trials)

    my_funcs.get_pics_and_pde_params(best_id,exp,exp_path)

if __name__ == '__main__':
        p=mp.Pool(processes = num_processors)
        try:
            p.map(all_func,experiments)
        except Exception as e:
            print(f'Failed with: {e}')
        p.close()
        p.join()