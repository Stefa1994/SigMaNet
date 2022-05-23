import os, sys

epochs = '3000'
for data in ['dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2', 'dataset_nodes500_alpha0.05_beta0.2'
            ]:
    for lr in [1e-3]:#, 1e-2, 5e-3]:
        # MagNet
        log_path = 'Signum_' + data
        for num_filter in [#16, 32, 
        64]:
                command = ('python3 node_SigNum.py ' 
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
                #command = ('python3 node_SigNum.py ' 
                #            +' --dataset='+data
                #            +' --num_filter='+str(num_filter)
                #            +' --K=1'
                #            +' --log_path='+str(log_path)
                #            +' --layer=2'
                #            +' --epochs='+epochs
                #            +' --dropout=0.5'
                #            +' --lr='+str(lr)
                #            #+' -N'
                #            +' -F' )
                #print(command)
                #os.system(command)
        
        