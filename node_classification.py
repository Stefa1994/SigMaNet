import os, sys

epochs = '3000'
for data in [#'telegram',  'dataset_nodes500_alpha0.05_beta0.2', 'dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2'
    #'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue.pk',
    #'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue.pk',
    #'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue.pk'
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue',
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue',
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue'
    ]:
    for lr in [1e-3]:
        # MagNet
        log_path = 'Magnet_' + data
        for num_filter in [#16, 32, 
        64]:
            for q in [0.01, 0.05, 0.1, 0.15, 0.2, 
            0.25]:
                command = ('python3 src/sparse_Magnet.py ' 
                            +' --dataset='+data
                            +' --q='+str(q)
                            +' --num_filter='+str(num_filter)
                            +' --K=1'
                            +' --log_path='+str(log_path)
                            +' --layer=2'
                            +' --epochs='+epochs
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' -a')
                print(command)
                os.system(command)
        # DGCN
        log_path = 'Sym_' + data
        for num_filter in [#5, 15, 
            30]:
            command = ('python3 src/Sym_DiGCN.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --lr='+str(lr)
                        +' --epochs='+epochs)
            print(command)
            os.system(command)
                
        log_path = 'GCN_' + data
        for num_filter in [#16, 32, 
        64]:
            command = ('python3 src/GCN.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --epochs='+epochs
                        +' --lr='+str(lr)
                        +' -tud')
            print(command)
            os.system(command)

        log_path = 'Cheb_' + data
        for num_filter in [#16, 32, 
        64]:
            command = ('python3 src/Cheb.py ' 
                        +' --dataset='+data
                        +' --K=2'
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --lr='+str(lr)
                        +' --epochs='+epochs
                        +' -tud')
            print(command)
            os.system(command)
#
        log_path = 'SAGE_' + data
        for num_filter in [#16, 32, 
        64]:
            command = ('python3 src/SAGE.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --lr='+str(lr)
                        +' --epochs='+epochs)
            print(command)
            os.system(command)

        log_path = 'GAT_' + data
        for heads in [2, 4, 
        8]:
            for num_filter in [#16, 32, 
            64]:
                command = ('python3 src/GAT.py ' 
                            +' --dataset='+data
                            +' --heads='+str(heads)
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' --epochs='+epochs)
                print(command)
                os.system(command)


        log_path = 'GIN_' + data
        for num_filter in [#16, 32, 
        64]:
            command = ('python3 src/GIN.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --lr='+str(lr)
                        +' --epochs='+epochs)
            print(command)
            os.system(command)

        
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
                            +' --direction'
                            +' --weight_decay=0.0005'
                            +' --w_pbrc=1')
            print(command)
            os.system(command)


        
    