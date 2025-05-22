import pandas as pd
import numpy as np
import my_funcs

nb_cases=10
size_l = [60]*nb_cases
t = 25
dt = 0.001
out_samples=[1.0]*nb_cases
snr_l = [1,10,25,30,35,100]
sparsity_l = [10,30,40,50,80]
equil_order_l = [-0.5,-1.0,-1.6,-1.8,-2.0,-2.2]
for exp_type in ['noise','sparse','equil']:
    for case in range(1,nb_cases+1):
        size = size_l[case-1]
        fs_list = my_funcs.pde_rd_for_data(size,T=t,dt=dt,out_sample=out_samples[case-1])
        fsl_shape = fs_list.shape
        
        if exp_type == 'equil':
            for equil_order in equil_order_l:

                cell = fs_list.transpose((0,2,3,1))[0:-1,1:-1,1:-1,:].copy()

                top = fs_list.transpose((0,2,3,1))[0:-1,0:-2,1:-1,:].copy()
                bottom = fs_list.transpose((0,2,3,1))[0:-1,2:,1:-1,:].copy()
                left = fs_list.transpose((0,2,3,1))[0:-1,1:-1,0:-2,:].copy()
                right = fs_list.transpose((0,2,3,1))[0:-1,1:-1,2:,:].copy()

                neigh = np.concatenate((top,bottom,left,right),axis=-1)

                future = fs_list.transpose((0,2,3,1))[1:,1:-1,1:-1,:].copy()

                # Add Equillibrium vicinity
                x_dot = future-cell
                for eq in range(future.shape[0]):
                    # print(np.abs(x_dot[eq,:,:,0]).mean())
                    if np.abs(x_dot[eq,:,:,0]).mean()<10**equil_order:
                        break

                cell = cell[eq:,:,:,:].copy()
                neigh = neigh[eq:,:,:,:].copy()
                future = future[eq:,:,:,:].copy()

                cell = cell.reshape(((t-eq)*(size-2)*(size-2),2))
                neigh = neigh.reshape(((t-eq)*(size-2)*(size-2),8))
                future = future.reshape(((t-eq)*(size-2)*(size-2),2))

                x_dot = future-cell

                np.savez_compressed('Experiments/equil_1e{}/sim_data/training/training{}'.format(equil_order,case),cell=cell,neigh=neigh,future=future)
        elif exp_type == 'sparse':
            for sparsity in sparsity_l:

                cell = fs_list.transpose((0,2,3,1))[0:-1,1:-1,1:-1,:].copy()

                top = fs_list.transpose((0,2,3,1))[0:-1,0:-2,1:-1,:].copy()
                bottom = fs_list.transpose((0,2,3,1))[0:-1,2:,1:-1,:].copy()
                left = fs_list.transpose((0,2,3,1))[0:-1,1:-1,0:-2,:].copy()
                right = fs_list.transpose((0,2,3,1))[0:-1,1:-1,2:,:].copy()

                neigh = np.concatenate((top,bottom,left,right),axis=-1)

                future = fs_list.transpose((0,2,3,1))[1:,1:-1,1:-1,:].copy()
                
                # Add Temporal sparsity
                n_samples = int(future.shape[0]*(sparsity/100.0))
                sample_indices = np.random.randint(0,future.shape[0],n_samples)
                future = np.take(future,sample_indices,axis=0)
                cell = np.take(cell,sample_indices,axis=0)
                neigh = np.take(neigh,sample_indices,axis=0)

                cell = cell.reshape((n_samples*(size-2)*(size-2),2))
                neigh = neigh.reshape((n_samples*(size-2)*(size-2),8))
                future = future.reshape((n_samples*(size-2)*(size-2),2))
                
                x_dot = future-cell

                np.savez_compressed('Experiments/sparse{}/sim_data/training/training{}'.format(sparsity,case),cell=cell,neigh=neigh,future=future)
        elif exp_type == 'noise':
            for snr in snr_l:

                cell = fs_list.transpose((0,2,3,1))[0:-1,1:-1,1:-1,:].copy()

                top = fs_list.transpose((0,2,3,1))[0:-1,0:-2,1:-1,:].copy()
                bottom = fs_list.transpose((0,2,3,1))[0:-1,2:,1:-1,:].copy()
                left = fs_list.transpose((0,2,3,1))[0:-1,1:-1,0:-2,:].copy()
                right = fs_list.transpose((0,2,3,1))[0:-1,1:-1,2:,:].copy()

                neigh = np.concatenate((top,bottom,left,right),axis=-1)

                future = fs_list.transpose((0,2,3,1))[1:,1:-1,1:-1,:].copy()
                
                # Add Gaussian noise
                for i in range(future.shape[0]):
                    cell[i,:,:,0] += np.random.normal(0,np.sqrt(cell[i,:,:,0].mean()*10**(-snr/10)),cell[i,:,:,0].shape)
                    cell[i,:,:,1] += np.random.normal(0,np.sqrt(cell[i,:,:,1].mean()*10**(-snr/10)),cell[i,:,:,1].shape)

                    neigh[i,:,:,0] += np.random.normal(0,np.sqrt(neigh[i,:,:,0].mean()*10**(-snr/10)),neigh[i,:,:,0].shape)
                    neigh[i,:,:,1] += np.random.normal(0,np.sqrt(neigh[i,:,:,1].mean()*10**(-snr/10)),neigh[i,:,:,1].shape)
                    neigh[i,:,:,2] += np.random.normal(0,np.sqrt(neigh[i,:,:,2].mean()*10**(-snr/10)),neigh[i,:,:,2].shape)
                    neigh[i,:,:,3] += np.random.normal(0,np.sqrt(neigh[i,:,:,3].mean()*10**(-snr/10)),neigh[i,:,:,3].shape)
                    neigh[i,:,:,4] += np.random.normal(0,np.sqrt(neigh[i,:,:,4].mean()*10**(-snr/10)),neigh[i,:,:,4].shape)
                    neigh[i,:,:,5] += np.random.normal(0,np.sqrt(neigh[i,:,:,5].mean()*10**(-snr/10)),neigh[i,:,:,5].shape)
                    neigh[i,:,:,6] += np.random.normal(0,np.sqrt(neigh[i,:,:,6].mean()*10**(-snr/10)),neigh[i,:,:,6].shape)
                    neigh[i,:,:,7] += np.random.normal(0,np.sqrt(neigh[i,:,:,7].mean()*10**(-snr/10)),neigh[i,:,:,7].shape)

                    future[i,:,:,0] += np.random.normal(0,np.sqrt(future[i,:,:,0].mean()*10**(-snr/10)),future[i,:,:,0].shape)
                    future[i,:,:,1] += np.random.normal(0,np.sqrt(future[i,:,:,1].mean()*10**(-snr/10)),future[i,:,:,1].shape)
                
                cell = cell.reshape(((fsl_shape[0]-1)*(size-2)*(size-2),2))
                neigh = neigh.reshape(((fsl_shape[0]-1)*(size-2)*(size-2),8))
                future = future.reshape(((fsl_shape[0]-1)*(size-2)*(size-2),2))

                x_dot = future-cell

                np.savez_compressed('Experiments/snr{}/sim_data/training/training{}'.format(snr,case),cell=cell,neigh=neigh,future=future)
        else:
            print('Wrong experiment name defined!')
            break