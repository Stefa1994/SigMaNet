import os, sys

epochs = '3000'
for data in [#'telegram/telegram' #,  'dataset_nodes500_alpha0.05_beta0.2', 'dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2'
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue',
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue',
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue'
    #'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue.pk',
    #'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue.pk',
    #'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue.pk'
]:
    for lr in [1e-3]:
        # SigMaNet
        log_path = 'SigMaNet_' + data
        for num_filter in [#16, 32,  
        64]:
                command = ('python3 src/node_SigMaNet.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --K=1'
                            +' --log_path='+str(log_path)
                            +' --layer=2'
                            +' --epochs='+epochs
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' -N'
                            +' -F' )
                print(command)
                os.system(command)
        # K=10 following the original paper
        log_path = 'APPNP_' + data
        for num_filter in [#16, 32, 
        64]:
            for alpha in [0.05, 0.1, 0.15, 
            0.2]: 
                command = ('python3 src/APPNP.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --alpha='+str(alpha)
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' --epochs='+epochs)
                print(command)
                os.system(command)

        
        log_path = 'DiGraph_' + data
        for num_filter in [#16, 32, 
        64]:
            for alpha in [0.05, 0.1, 0.15, 
            0.2]: 
                command = ('python3 src/Digraph.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --alpha='+str(alpha)
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' --epochs='+epochs)
                print(command)
                os.system(command)
        
        