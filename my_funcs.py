import numpy as np
import pandas as pd
import json
from Experiments.snr100 import network as net
from skimage.metrics import structural_similarity
import cv2
from sklearn import linear_model
import matplotlib.pyplot as plt
import multiprocessing as mp

def laplacian(X, dx):
    top = X[0:-2, 1:-1]
    left = X[1:-1, 0:-2]
    bottom = X[2:, 1:-1]
    right = X[1:-1, 2:]
    center = X[1:-1, 1:-1]
    return (top + left + bottom + right - 4 * center) / dx**2

def pde_rd_for_data(size=100,T=20.0,dt=0.001,out_sample=1):
    a = 2.8e-4
    b = 5e-3
    tau = .1
    k = -.005

    dx = 2. / size  # space step
    n = int(T / dt)  # number of iterations
    fs_list = []

    u = np.random.rand(size, size)
    v = np.random.rand(size, size)
    for z in (u, v):
            z[0, :] = z[1, :]
            z[-1, :] = z[-2, :]
            z[:, 0] = z[:, 1]
            z[:, -1] = z[:, -2]

    fs_list.append([u.copy(),v.copy()])

    # We simulate the PDE with the finite difference
    # method.
    

    for i in range(n):
        # We compute the Laplacian of u and v.
        deltaU = laplacian(u, dx)
        deltaV = laplacian(v, dx)

        # We take the values of u and v inside the grid.
        Uc = u[1:-1, 1:-1]
        Vc = v[1:-1, 1:-1]

        # We update the variables.
        u[1:-1, 1:-1] = Uc + dt * (a * deltaU + Uc - Uc**3 - Vc + k)
        v[1:-1, 1:-1] = Vc + dt * (b * deltaV + Uc - Vc) / tau
        
        # Neumann conditions: derivatives at the edges
        # are null.
        for z in (u, v):
            z[0, :] = z[1, :]
            z[-1, :] = z[-2, :]
            z[:, 0] = z[:, 1]
            z[:, -1] = z[:, -2]

        # We sample the state of the system
        if (i+1)%int(out_sample/dt)==0:
            fs_list.append([u.copy(),v.copy()])

    return np.asarray(fs_list)

def my_pde_rd(u,v,size=100,T=20.0,dt=0.001,out_sample=1):
    a = 2.8e-4
    b = 5e-3
    tau = .1
    k = -.005

    dx = 2. / size  # space step
    n = int(T / dt)  # number of iterations
    fs_list = []

    for z in (u, v):
            z[0, :] = z[1, :]
            z[-1, :] = z[-2, :]
            z[:, 0] = z[:, 1]
            z[:, -1] = z[:, -2]

    fs_list.append([u.copy(),v.copy()])

    for i in range(n):
        # We compute the Laplacian of u and v.
        deltaU = laplacian(u, dx)
        deltaV = laplacian(v, dx)
        # We take the values of u and v inside the grid.
        Uc = u[1:-1, 1:-1]
        Vc = v[1:-1, 1:-1]
        # We update the variables.
        u[1:-1, 1:-1], v[1:-1, 1:-1] = \
            Uc + dt * (a * deltaU + Uc - Uc**3 - Vc + k),\
            Vc + dt * (b * deltaV + Uc - Vc) / tau
        # Neumann conditions: derivatives at the edges
        # are null.
        for z in (u, v):
            z[0, :] = z[1, :]
            z[-1, :] = z[-2, :]
            z[:, 0] = z[:, 1]
            z[:, -1] = z[:, -2]


        if (i+1)%int(out_sample/dt)==0:
            fs_list.append([u.copy(),v.copy()])


    return np.asarray(fs_list)

## Histogram based image compare
def hist_img_comp(img1, img2):
    bins = 10
    return np.round(cv2.compareHist(np.asarray(np.histogram(img1,range=(-1,1),bins=bins)[0]).reshape(bins,1).astype('float32'), np.asarray(np.histogram(img2,range=(-1,1),bins=bins)[0]).reshape(bins,1).astype('float32'),cv2.HISTCMP_CORREL),3)

def show_patterns(tag,u,tstep,snr):
    plt.figure(figsize=(8, 8))
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if tag=='sim':
        try:    
            plt.imshow(u, cmap=plt.cm.copper,
                    interpolation='bilinear',
                    extent=[-1, 1, -1, 1])
            plt.axis('off')
            plt.savefig('./Visualizations/{}/simresult_tstep{}.png'.format(snr,tstep))
        except:
            print("Errors!!!!")
    elif tag=='obs':
        try:
            plt.imshow(u, cmap=plt.cm.copper,
                    interpolation='bilinear',
                    extent=[-1, 1, -1, 1])
            plt.axis('off')
            plt.savefig('./Visualizations/{}/obsresult_tstep{}.png'.format(snr,tstep))
        except:
            print("Errors!!!!")
    else:
        print('Wrong tag!!!!')
        
    plt.clim(-1,1)

def get_mech_param(fs_list,dx,library,reg='lasso'):
    
    delta_uv = fs_list[1:,:,1:-1, 1:-1] - fs_list[0:-1,:,1:-1, 1:-1]
    delta_u = delta_uv[:,0,np.newaxis,:,:].reshape(-1)[:,np.newaxis]
    delta_v = delta_uv[:,1,np.newaxis,:,:].reshape(-1)[:,np.newaxis]

    temp_u_v = fs_list[0:-1,:,:,:]
    temp_u_center = temp_u_v[:,0,np.newaxis,1:-1, 1:-1].reshape(-1)[:,np.newaxis]
    temp_v_center = temp_u_v[:,1,np.newaxis,1:-1, 1:-1].reshape(-1)[:,np.newaxis]
    temp_lapl_u_v = (temp_u_v[:,:,0:-2, 1:-1] + temp_u_v[:,:,1:-1, 0:-2] + temp_u_v[:,:,2:, 1:-1] + temp_u_v[:,:,1:-1, 2:] - 4 * temp_u_v[:,:,1:-1, 1:-1]) / dx**2
    temp_lapl_u = temp_lapl_u_v[:,0,np.newaxis,:,:].reshape(-1)[:,np.newaxis]
    temp_lapl_v = temp_lapl_u_v[:,1,np.newaxis,:,:].reshape(-1)[:,np.newaxis]

    l = []
    for i in library: l.append(i(temp_u_center,temp_v_center))

    theta = np.concatenate((temp_lapl_u, temp_lapl_v, np.asarray(l).transpose(1,0,2)[:,:,0]), axis=-1)

    if reg=='lasso':
        param_u_model = linear_model.Lasso(alpha=5e-4,fit_intercept=False,max_iter=100000,tol=1e-6)
        param_v_model = linear_model.Lasso(alpha=5e-4,fit_intercept=False,max_iter=100000,tol=1e-6)
    else:
        param_u_model = linear_model.Ridge(alpha=1.0,fit_intercept=False,max_iter=100000)
        param_v_model = linear_model.Ridge(alpha=1.0,fit_intercept=False,max_iter=100000)

    param_u_model.fit(theta,delta_u)

    param_v_model.fit(theta,delta_v)

    return param_u_model, param_v_model

def besttrial_reruns_unpack(args):
    return besttrial_reruns(*args)

def besttrial_reruns(i,snr,best_trial):
    exp_path = './Experiments/{}/'.format(snr)
    size = 60
    dt = 0.001
    out_sample=1.0
    tsteps = 8
    
    combined_metrics = {}
    u = np.random.rand(size, size)
    v = np.random.rand(size, size)

    sim_uv = np.concatenate((u[:,:,np.newaxis],v[:,:,np.newaxis]),axis=2)

    sim_uv[0, :, :] = sim_uv[1, :, :]
    sim_uv[-1, :, :] = sim_uv[-2, :, :]
    sim_uv[:, 0, :] = sim_uv[:, 1, :]
    sim_uv[:, -1, :] = sim_uv[:, -2, :]
    sim_uv_obs = sim_uv.copy()

    fs_obs_list = my_pde_rd(sim_uv_obs[:,:,0],sim_uv_obs[:,:,1],size,tsteps,dt,out_sample=out_sample)


    try:
        f = open(exp_path+'tmp/untitled_project/'+'{}/trial.json'.format(best_trial))
        curr_iter = json.load(f)
        curr_parameters = pd.DataFrame(curr_iter['hyperparameters']['values'], index=[0])
    except:
        print('Error')

    params={}

    ### Experiment Parameters ###
    params['state_features'] = 2

    ### Hyper Parameters ###
    params['hp_num_neighbors'] = 4

    params['hp_latent_space'] = curr_parameters['hp_latent_space'].values[0]

    params['hp_beta_units'] = curr_parameters['hp_beta_units'].values[0]

    enc_dec_size = curr_parameters['hp_enc_dec_size'].values[0]
    params['hp_phi_enc_units'] = enc_dec_size
    params['hp_psi_enc_units'] = enc_dec_size
    params['hp_psi_dec_units'] = enc_dec_size

    params['hp_delta_units'] = curr_parameters['hp_delta_units'].values[0]

    params['hp_dropout'] = curr_parameters['hp_dropout'].values[0]

    ae, pipeline = net.pipeline(params)

    pipeline.load_weights(exp_path+"Realizations/"+best_trial+"_pipeline.h5")
    
    exp_data_path = exp_path+'sim_data/training/'

    x_dot_abs_0_max = np.load(exp_data_path+'x_dot_abs_0_max.npy')
    x_dot_abs_1_max = np.load(exp_data_path+'x_dot_abs_1_max.npy')

    
    fs_sim_list=[]
    fs_sim_list.append(sim_uv.copy())

    sim_uv_sim = sim_uv.copy()

    def simulate_rd(size):
        cell = sim_uv_sim[1:-1,1:-1,:].reshape((size-2)**2,2)

        top = sim_uv_sim[0:-2,1:-1,:].reshape((size-2)**2,2)
        bottom = sim_uv_sim[2:,1:-1,:].reshape((size-2)**2,2)
        left = sim_uv_sim[1:-1,0:-2,:].reshape((size-2)**2,2)
        right = sim_uv_sim[1:-1,2:,:].reshape((size-2)**2,2)

        neigh = np.concatenate((top,bottom,left,right),axis=1)

        future_dot = pipeline.predict([cell,neigh])
        future_dot[:,0] = future_dot[:,0] * x_dot_abs_0_max
        future_dot[:,1] = future_dot[:,1] * x_dot_abs_1_max

        future = cell + future_dot

        sim_uv_sim[1:-1,1:-1,:] = future.reshape((size-2),(size-2),2)

        sim_uv_sim[0, :, :] = sim_uv_sim[1, :, :]
        sim_uv_sim[-1, :, :] = sim_uv_sim[-2, :, :]
        sim_uv_sim[:, 0, :] = sim_uv_sim[:, 1, :]
        sim_uv_sim[:, -1, :] = sim_uv_sim[:, -2, :]

        fs_sim_list.append(sim_uv_sim.copy())

    for t in range(int(tsteps)):
        simulate_rd(size)

    mae_l=[]
    for j in range(0,(tsteps+1)):
        # Compute MAE between two images
        mae = np.abs((np.asarray(fs_sim_list)[j,:,:,0] - fs_obs_list[j,0,:,:])).mean()
        mae_l.append(mae)

    ssim_l = []
    for k in range(0,(tsteps+1)):
        # Compute SSIM between two images
        (score, diff) = structural_similarity(fs_obs_list[k][0,:,:], fs_sim_list[k][:,:,0], full=True)
        ssim_l.append(score)        

    hist_l = []
    for l in range(0,(tsteps+1)):
        # Compute HIST between two images
        score_hist = hist_img_comp(fs_obs_list[l][0,:,:], fs_sim_list[l][:,:,0])
        hist_l.append(score_hist)
    
    combined_metrics[str(i)] = {'mae':mae_l[-1],'ssim':ssim_l[-1],'hist':hist_l[-1]}

    return combined_metrics

def parallel_runs(num_reruns,num_processors,snr,best_id):
        p=mp.Pool(processes = num_processors)
        try:
            results = [p.apply_async(besttrial_reruns,args=(i,snr,best_id,)) for i in range(num_reruns)]
        except Exception as e:
            print(f'Failed with: {e}')
        p.close()
        p.join()

        out = [result.get() for result in results]
        print(out)

        return out

def get_best_trial(snr,exp_path,num_trials):
    combined_metrics = {}
    size = 60
    dt = 0.001
    out_sample=1.0
    tsteps = 8

    u = np.random.rand(size, size)
    v = np.random.rand(size, size)

    sim_uv = np.concatenate((u[:,:,np.newaxis],v[:,:,np.newaxis]),axis=2)

    sim_uv[0, :, :] = sim_uv[1, :, :]
    sim_uv[-1, :, :] = sim_uv[-2, :, :]
    sim_uv[:, 0, :] = sim_uv[:, 1, :]
    sim_uv[:, -1, :] = sim_uv[:, -2, :]
    sim_uv_obs = sim_uv.copy()
    # fs_obs_list = pde_rd_grayscott(sim_uv_obs[:,:,0],sim_uv_obs[:,:,1],size,tsteps,dt,out_sample=out_sample)
    fs_obs_list = my_pde_rd(sim_uv_obs[:,:,0],sim_uv_obs[:,:,1],size,tsteps,dt,out_sample=out_sample)

    for i in range(num_trials):
        # print('Analyzing Trial Number {}'.format(i))
        if i > 9:
            try:
                curr_id = 'trial_{}'.format(str(i))
                f = open(exp_path+'tmp/untitled_project/'+'trial_{}/trial.json'.format(str(i)))
                curr_iter = json.load(f)
                curr_parameters = pd.DataFrame(curr_iter['hyperparameters']['values'], index=[0])
            except:
                print('Error')
        else:
            try:
                curr_id = 'trial_0{}'.format(str(i))
                f = open(exp_path+'tmp/untitled_project/'+'trial_0{}/trial.json'.format(str(i)))
                curr_iter = json.load(f)
                curr_parameters = pd.DataFrame(curr_iter['hyperparameters']['values'], index=[0])
            except:
                print('Error')

        params={}

        ### Experiment Parameters ###
        params['state_features'] = 2

        ### Hyper Parameters ###
        params['hp_num_neighbors'] = 4

        params['hp_latent_space'] = curr_parameters['hp_latent_space'].values[0]

        params['hp_beta_units'] = curr_parameters['hp_beta_units'].values[0]

        enc_dec_size = curr_parameters['hp_enc_dec_size'].values[0]
        params['hp_phi_enc_units'] = enc_dec_size
        params['hp_psi_enc_units'] = enc_dec_size
        params['hp_psi_dec_units'] = enc_dec_size

        params['hp_delta_units'] = curr_parameters['hp_delta_units'].values[0]

        # l2_reg = curr_parameters['hp_l2_reg'].values[0]
        # # params['hp_l1_reg'] = l1_reg
        # params['hp_l2_reg'] = l2_reg

        params['hp_dropout'] = curr_parameters['hp_dropout'].values[0]

        ae, pipeline = net.pipeline(params)
        try:
            pipeline.load_weights(exp_path+"Realizations/"+curr_id+"_pipeline.h5")
            # pipeline.load_weights("Experiments/Realizations/"+snr+"/"+curr_id+"_pipeline.h5")
        except:
            continue
        
        exp_data_path = exp_path+'sim_data/training/'#+snr+"/"


        x_dot_abs_0_max = np.load(exp_data_path+'x_dot_abs_0_max.npy')
        x_dot_abs_1_max = np.load(exp_data_path+'x_dot_abs_1_max.npy')

        
        fs_sim_list=[]
        fs_sim_list.append(sim_uv.copy())

        sim_uv_sim = sim_uv.copy()

        def simulate_rd(size):
            cell = sim_uv_sim[1:-1,1:-1,:].reshape((size-2)**2,2)

            top = sim_uv_sim[0:-2,1:-1,:].reshape((size-2)**2,2)
            bottom = sim_uv_sim[2:,1:-1,:].reshape((size-2)**2,2)
            left = sim_uv_sim[1:-1,0:-2,:].reshape((size-2)**2,2)
            right = sim_uv_sim[1:-1,2:,:].reshape((size-2)**2,2)

            neigh = np.concatenate((top,bottom,left,right),axis=1)

            # neigh_lapl = (np.sum(np.concatenate((top[:,np.newaxis,:],bottom[:,np.newaxis,:],left[:,np.newaxis,:],right[:,np.newaxis,:]),axis=1),axis=1)-4*cell)/8
            # neigh = np.concatenate((top,bottom,left,right,neigh_lapl),axis=1)

            future_dot = pipeline.predict([cell,neigh])
            future_dot[:,0] = future_dot[:,0] * x_dot_abs_0_max
            future_dot[:,1] = future_dot[:,1] * x_dot_abs_1_max

            # print(future_dot[:,0].mean())
            future = cell + future_dot
            # for h in range(future.shape[1]):
            #     future[:,h] = np.where(future[:,h]>1,1,future[:,h])
            #     future[:,h] = np.where(future[:,h]<0,0,future[:,h])
            sim_uv_sim[1:-1,1:-1,:] = future.reshape((size-2),(size-2),2)

            sim_uv_sim[0, :, :] = sim_uv_sim[1, :, :]
            sim_uv_sim[-1, :, :] = sim_uv_sim[-2, :, :]
            sim_uv_sim[:, 0, :] = sim_uv_sim[:, 1, :]
            sim_uv_sim[:, -1, :] = sim_uv_sim[:, -2, :]

            fs_sim_list.append(sim_uv_sim.copy())

        for t in range(int(tsteps)):
            # print('Timestep: {}'.format(int((t+1))))
            simulate_rd(size)

        mae_l=[]
        for i in range(0,(tsteps+1)):
            # Compute MAE between two images
            mae = np.abs((np.asarray(fs_sim_list)[i,:,:,0] - fs_obs_list[i,0,:,:])).mean()
            # print("Image similarity", score)
            mae_l.append(mae)

        ssim_l = []
        for i in range(0,(tsteps+1)):
            # Compute SSIM between two images
            (score, diff) = structural_similarity(fs_obs_list[i][0,:,:], fs_sim_list[i][:,:,0], full=True)
            # print("Image similarity", score)
            ssim_l.append(score)        

        hist_l = []
        for i in range(0,(tsteps+1)):
            # Compute HIST between two images
            score_hist = hist_img_comp(fs_obs_list[i][0,:,:], fs_sim_list[i][:,:,0])
            # print("Image similarity", score)
            hist_l.append(score_hist)
        
        combined_metrics[curr_id] = {'mae':mae_l[-1],'ssim':ssim_l[-1],'hist':hist_l[-1]}

    with open(exp_path+"combined_metrics_{}.json".format(snr), "w") as fp:
        json.dump(combined_metrics, fp)

    metric = 'mae'
    best = 1
    for item in combined_metrics.items():
        key, value = item
        if value[metric] < best:
            best = value[metric]
            best_id = key

    with open(exp_path+"besttrial_{}.txt".format(snr), "w") as fp2:
        fp2.writelines([best_id])

    return best_id

def get_pics_and_pde_params(best_id,snr,exp_path):
    size = 60
    dt = 0.001
    out_sample = 1.0
    tsteps = 8

    init_uv = np.random.rand(size, size,2)

    init_uv[0, :, :] = init_uv[1, :, :]
    init_uv[-1, :, :] = init_uv[-2, :, :]
    init_uv[:, 0, :] = init_uv[:, 1, :]
    init_uv[:, -1, :] = init_uv[:, -2, :]
    sim_uv_obs = init_uv.copy()

    # fs_obs_list = pde_rd_grayscott(sim_uv_obs[:,:,0],sim_uv_obs[:,:,1],size=size,T=tsteps,dt=dt,out_sample=out_sample)
    fs_obs_list = my_pde_rd(sim_uv_obs[:,:,0],sim_uv_obs[:,:,1],size=size,T=tsteps,dt=dt,out_sample=out_sample)

    curr_id = best_id# 
    f = open(exp_path+'tmp/untitled_project/'+'{}/trial.json'.format(curr_id))
    curr_iter = json.load(f)
    curr_parameters = pd.DataFrame(curr_iter['hyperparameters']['values'], index=[0])

    params={}

    ### Experiment Parameters ###
    params['state_features'] = 2

    ### Hyper Parameters ###
    params['hp_num_neighbors'] = 4

    params['hp_latent_space'] = curr_parameters['hp_latent_space'].values[0]

    params['hp_beta_units'] = curr_parameters['hp_beta_units'].values[0]

    enc_dec_size = curr_parameters['hp_enc_dec_size'].values[0]
    params['hp_phi_enc_units'] = enc_dec_size
    params['hp_psi_enc_units'] = enc_dec_size
    params['hp_psi_dec_units'] = enc_dec_size

    params['hp_delta_units'] = curr_parameters['hp_delta_units'].values[0]

    # l2_reg = curr_parameters['hp_l2_reg'].values[0]
    # params['hp_l2_reg'] = l2_reg

    params['hp_dropout'] = curr_parameters['hp_dropout'].values[0]

    ae, pipeline = net.pipeline(params)
    pipeline.load_weights(exp_path+"Realizations/"+curr_id+"_pipeline.h5")
    # pipeline.load_weights("Experiments/Realizations/"+snr+"/"+curr_id+"_pipeline.h5")

    exp_data_path = exp_path+'sim_data/training/'#+snr+"/"

    x_dot_abs_0_max = np.load(exp_data_path+'x_dot_abs_0_max.npy')
    x_dot_abs_1_max = np.load(exp_data_path+'x_dot_abs_1_max.npy')

    fs_sim_list=[]
    fs_sim_list.append(init_uv.copy())

    sim_uv = init_uv.copy()

    def simulate_rd(size):
        cell = sim_uv[1:-1,1:-1,:].reshape((size-2)**2,2)

        top = sim_uv[0:-2,1:-1,:].reshape((size-2)**2,2)
        bottom = sim_uv[2:,1:-1,:].reshape((size-2)**2,2)
        left = sim_uv[1:-1,0:-2,:].reshape((size-2)**2,2)
        right = sim_uv[1:-1,2:,:].reshape((size-2)**2,2)

        neigh = np.concatenate((top,bottom,left,right),axis=1)

        # neigh_lapl = (np.sum(np.concatenate((top[:,np.newaxis,:],bottom[:,np.newaxis,:],left[:,np.newaxis,:],right[:,np.newaxis,:]),axis=1),axis=1)-4*cell)/8
        # neigh = np.concatenate((top,bottom,left,right,neigh_lapl),axis=1)

        future_dot = pipeline.predict([cell,neigh])
        future_dot[:,0] = future_dot[:,0] * x_dot_abs_0_max
        future_dot[:,1] = future_dot[:,1] * x_dot_abs_1_max

        # print(future_dot[:,0].mean())
        future = cell + future_dot
        # for h in range(future.shape[1]):
        #     future[:,h] = np.where(future[:,h]>1,1,future[:,h])
        #     future[:,h] = np.where(future[:,h]<0,0,future[:,h])
        sim_uv[1:-1,1:-1,:] = future.reshape((size-2),(size-2),2)

        sim_uv[0, :, :] = sim_uv[1, :, :]
        sim_uv[-1, :, :] = sim_uv[-2, :, :]
        sim_uv[:, 0, :] = sim_uv[:, 1, :]
        sim_uv[:, -1, :] = sim_uv[:, -2, :]

        fs_sim_list.append(sim_uv.copy())

    for t in range(int(tsteps/out_sample)):
        # print('Timestep: {}'.format(int((t+1))))
        simulate_rd(size)


    for i in range(0,int(tsteps/out_sample+1)):
        if (i+1)%int(1/out_sample)==0:
            show_patterns('sim',fs_sim_list[i][:,:,0],int(i*out_sample),snr)

    for i in range(0,int(tsteps/out_sample+1)):
        if (i+1)%int(1/out_sample)==0:
            show_patterns('obs',fs_obs_list[i][0,:,:],int(i*out_sample),snr)

    library = [
        lambda x,y : np.ones((x.shape[0],1)),
        lambda x,y : x,
        lambda x,y : y,
        lambda x,y : np.square(x),
        lambda x,y : np.square(y),
        lambda x,y : x*y,
        lambda x,y : np.square(x)*y,
        lambda x,y : x*np.square(y),
        lambda x,y : np.square(x)*np.square(y),
        lambda x,y : np.power(x,3),
        lambda x,y : np.power(y,3),
        lambda x,y : np.power(y,3)*x,
        lambda x,y : np.power(x,3)*y,
        lambda x,y : np.power(x,3)*np.square(y),
        lambda x,y : np.power(y,3)*np.square(x),
        lambda x,y : np.power(x,3)*np.power(y,3)
    ]

    size = 60
    dt = 0.001
    out_sample = 1
    tsteps = 8

    sim_uv_obs = init_uv.copy()

    fs_obs_list = my_pde_rd(sim_uv_obs[:,:,0],sim_uv_obs[:,:,1],size=size,T=tsteps,dt=dt,out_sample=out_sample)

    out_samples=out_sample
    reg_type = 'lasso'
    param_u_obs, param_v_obs = get_mech_param(fs_obs_list,2./size,library,reg=reg_type)
    param_u_sim, param_v_sim = get_mech_param(np.asarray(fs_sim_list).transpose(0,3,1,2),2./size,library,reg=reg_type)

    param_table = pd.DataFrame(columns=['Parameters','U_origin','U_obs','U_sim','V_origin','V_obs','V_sim'])
    param_table['Parameters']=['del_U', 'del_V','1', 'U', 'V', 'U^2', 'V^2', 'UV', 'U^2V', 'UV^2', 'U^2V^2', 'U^3', 'V^3','UV^3','U^3V','U^3V^2','U^2V^3','U^3V^3']
    param_table['U_obs'] = np.asarray(param_u_obs.coef_).flatten()/out_samples
    param_table['U_sim'] = np.asarray(param_u_sim.coef_).flatten()/out_samples
    param_table['V_obs'] = np.asarray(param_v_obs.coef_).flatten()/out_samples
    param_table['V_sim'] = np.asarray(param_v_sim.coef_).flatten()/out_samples
    param_table['U_origin'] = [2.8e-4,0,5e-3,1,-1,0,0,0,0,0,0,-1,0,0,0,0,0,0]
    param_table['V_origin'] = [0,5e-2,0,10,-10,0,0,0,0,0,0,0,0,0,0,0,0,0]
    param_table = param_table.fillna(0)
    param_table.to_csv('./Visualizations/{}/PDE_param.csv'.format(snr),index=False)


def get_exp_design(snr,exp_path,num_trials):
    parameters = pd.DataFrame()
    for i in range(num_trials):
        if i<10:
            best_id='trial_0{}'.format(i)
        else:
            best_id='trial_{}'.format(i)
        try:
            f = open(exp_path+'tmp/untitled_project/'+best_id+'/trial.json')
        except:
            continue
        best_iter = json.load(f)
        best_parameters = pd.DataFrame(best_iter['hyperparameters']['values'], index=[0])
        best_parameters['trial_id'] = i
        first_column = best_parameters.pop('trial_id')
        best_parameters.insert(0, 'trial_id', first_column)
        parameters = parameters.append(best_parameters)
    parameters.to_csv(exp_path+'experimental_design_{}.csv'.format(snr),index=False)