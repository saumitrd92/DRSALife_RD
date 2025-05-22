import pandas as pd

#####################################################
####### Below combines all experiment results #######
#####################################################

df_combined = pd.DataFrame(columns=['experiment','trial','mae','ssim','hist'])

experiments = ['equil_1e-0.5','equil_1e-1.0','equil_1e-1.6','equil_1e-1.8','equil_1e-2.0','equil_1e-2.2','snr1','snr10','snr25','snr30','snr35','snr100','sparse10','sparse30','sparse40','sparse50','sparse80']
for snr in experiments:


    df_temp = pd.read_csv('./Experiments/'+snr+'/combined_besttrial_reruns_'+snr+'.csv')
    df_temp = df_temp.rename(columns={'Unnamed: 0':'trial'})
    df_temp['experiment'] = [snr]*df_temp.shape[0]
    df_combined = df_combined.append(df_temp)

df_combined.to_csv('./Visualizations/Combined_results.csv')