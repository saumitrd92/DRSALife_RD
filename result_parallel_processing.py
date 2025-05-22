import pandas as pd
import multiprocessing as mp
import my_funcs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

experiments = ['equil_1e-0.5','equil_1e-1.0','equil_1e-1.6','equil_1e-1.8','equil_1e-2.0','equil_1e-2.2','snr1','snr10','snr25','snr30','snr35','snr100','sparse10','sparse30','sparse40','sparse50','sparse80']
for snr in experiments:
    exp_path = './Experiments/{}/'.format(snr)
    num_trials = 50

    with open(exp_path+"besttrial_{}.txt".format(snr), "r") as fp2:
        best_id = fp2.readlines()[0]

    ###########################################################################
    ####### Below performs several test simulation using the best trial #######
    ###########################################################################

    # Test simulations
    num_reruns=100
    num_processors = 8

    if __name__ == '__main__':
        p=mp.Pool(processes = num_processors)
        try:
            result_list = p.map(my_funcs.besttrial_reruns_unpack,[[i,snr,best_id] for i in range(num_reruns)])
        except Exception as e:
            print(f'Failed with: {e}')
        p.close()
        p.join()

        print('### Experiment {} ###'.format(snr))
        # print(result_list)

        combined_metrics1={}
        for i in result_list:
            combined_metrics1.update(i)
        pd.DataFrame(combined_metrics1).transpose().to_csv(exp_path+"combined_besttrial_reruns_{}.csv".format(snr))