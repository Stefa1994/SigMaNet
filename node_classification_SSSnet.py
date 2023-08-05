import os, sys

epochs = '3000'
for data in [#'telegram',
    #'dataset_nodes500_alpha0.05_beta0.2', 'dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2'  
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue',
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue',
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue',
    'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue',
    'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue',
    'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue'

]:
    for lr in [1e-3]:
        log_path = 'SSSNet_' + data
        for num_filter in [64]:
            command = ('python3 src/SSSNET.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --lr='+str(lr)
                            +' --epochs='+epochs
                            +' --direction'
                            +' --weight_decay=0.0005')
            print(command)
            os.system(command)

            command = ('python3 src/SSSNET.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --lr='+str(lr)
                            +' --epochs='+epochs
                            #+' --direction'
                            +' --weight_decay=0.0005')
            print(command)
            os.system(command)

            command = ('python3 src/SSSNET.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --lr='+str(lr)
                            +' --epochs='+epochs
                            +' --direction'
                            +' --weight_decay=0.0005'
                            +' --w_pbrc=1')
            print(command)
            os.system(command)
            command = ('python3 src/SSSNET.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --lr='+str(lr)
                            +' --epochs='+epochs
                            #+' --direction'
                            +' --weight_decay=0.0005'
                            +' --w_pbrc=1')
            print(command)
            os.system(command)
    