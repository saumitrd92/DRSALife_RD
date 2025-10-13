import pandas as pd
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

for exp in experiments:
    exp_path = './Experiments/{}/'.format(exp)
    num_trials = config.get("num_trials", 50)

    with open(exp_path+"besttrial_{}.txt".format(exp), "r") as fp2:
        best_id = fp2.readlines()[0]

    ###########################################################################
    ####### Below performs several test simulation using the best trial #######
    ###########################################################################

    # Test simulations
    num_reruns=config.get("num_reruns", 100)
    num_processors = config.get("num_processors", 8)

    if __name__ == '__main__':
        p=mp.Pool(processes = num_processors)
        try:
            result_list = p.map(my_funcs.besttrial_reruns_unpack,[[i,exp,best_id] for i in range(num_reruns)])
        except Exception as e:
            print(f'Failed with: {e}')
        p.close()
        p.join()

        print('### Experiment {} ###'.format(exp))
        # print(result_list)

        combined_metrics1={}
        for i in result_list:
            combined_metrics1.update(i)
        pd.DataFrame(combined_metrics1).transpose().to_csv(exp_path+"combined_besttrial_reruns_{}.csv".format(exp))