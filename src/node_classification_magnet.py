import os

epochs = '3000'
for data in ['dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2', 'dataset_nodes500_alpha0.05_beta0.2'
            ]:
    for lr in [1e-3]:#, 1e-2, 5e-3]:
        # MagNet
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
     