import os, sys

epochs = '3000'
#for data in ['dataset_nodes500_alpha0.1_beta0.05', 'dataset_nodes500_alpha0.1_beta0.1', 
#            'dataset_nodes500_alpha0.1_beta0.15', 'dataset_nodes500_alpha0.1_beta0.25',
#            'dataset_nodes500_alpha0.1_beta0.3', 'dataset_nodes500_alpha0.1_beta0.35',
#            'dataset_nodes500_alpha0.1_beta0.4'
#            ]:
for data in ['telegram',  'dataset_nodes500_alpha0.05_beta0.2', 'dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2']:
    for lr in [1e-3]:
        # MagNet
        log_path = 'Magnet_' + data
        for num_filter in [16, 32, 
        64]:
            for q in [0.01, 0.05, 0.1, 0.15, 0.2, 
            0.25]:
                command = ('python3 sparse_Magnet.py ' 
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
        for num_filter in [5, 15, 30]:
            command = ('python3 Sym_DiGCN.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --lr='+str(lr)
                        +' --epochs='+epochs)
            print(command)
            os.system(command)
                
        log_path = 'GCN_' + data
        for num_filter in [16, 32, 
        64]:
            command = ('python3 GCN.py ' 
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
        for num_filter in [16, 32, 
        64]:
            command = ('python3 Cheb.py ' 
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
        for num_filter in [16, 32, 
        64]:
            command = ('python3 SAGE.py ' 
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
            for num_filter in [16, 32, 
            64]:
                command = ('python3 GAT.py ' 
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
        for num_filter in [16, 32, 
        64]:
            command = ('python3 GIN.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --lr='+str(lr)
                        +' --epochs='+epochs)
            print(command)
            os.system(command)


        # K=10 following the original paper
        log_path = 'APPNP_' + data
        for num_filter in [16, 32, 
        64]:
            for alpha in [0.05, 0.1, 0.15, 
            0.2]: 
                command = ('python3 APPNP.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --alpha='+str(alpha)
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' --epochs='+epochs)
                print(command)
                os.system(command)

        
        log_path = 'DiG_' + data
        for num_filter in [16, 32, 
        64]:
            for alpha in [0.05, 0.1, 0.15, 
            0.2]: 
                command = ('python3 Digraph.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --alpha='+str(alpha)
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' --epochs='+epochs)
                print(command)
                os.system(command)
    