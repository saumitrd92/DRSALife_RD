import pandas as pd
import os
import json

#####################################################
####### Below combines all experiment results #######
#####################################################

df_combined = pd.DataFrame(columns=['experiment','trial','mae','ssim','hist'])

if os.path.exists('config_params.json'):
    config = json.load(open('config_params.json'))

experiments = []

for equil_order in config.get("equil_order_l", [-0.5,-1.0,-1.6,-1.8,-2.0,-2.2]):
    exp_name = 'equil_1e{}'.format(equil_order)
    experiments.append(exp_name)

for sparsity in config.get("sparsity_l", [10,30,40,50,80]):
    exp_name = 'sparse{}'.format(sparsity)
    experiments.append(exp_name)

for snr in config.get("snr_l", [1,10,25,30,35,100]):
    exp_name = 'snr{}'.format(snr)
    experiments.append(exp_name)

for exp in experiments:
    df_temp = pd.read_csv('./Experiments/'+exp+'/combined_besttrial_reruns_'+exp+'.csv')
    df_temp = df_temp.rename(columns={'Unnamed: 0':'trial'})
    df_temp['experiment'] = [exp]*df_temp.shape[0]
    df_combined = df_combined.append(df_temp)

df_combined.to_csv('./Visualizations/Combined_results.csv')