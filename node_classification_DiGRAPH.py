import os, sys

epochs = '3000'
#for data in ['dataset_nodes500_alpha0.1_beta0.05', 'dataset_nodes500_alpha0.1_beta0.1', 
#            'dataset_nodes500_alpha0.1_beta0.15', 'dataset_nodes500_alpha0.1_beta0.25',
#            'dataset_nodes500_alpha0.1_beta0.3', 'dataset_nodes500_alpha0.1_beta0.35',
#            'dataset_nodes500_alpha0.1_beta0.4'
#            ]:
for data in [#'telegram',  
'dataset_nodes500_alpha0.05_beta0.2', 'dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2']:
    for lr in [1e-3]:
        # MagNet
      
        log_path = 'DiG_' + data
        for num_filter in [#16, 32, 
        21]:
            for alpha in [0.05, 0.1, 0.15, 
            0.2]: 
                command = ('python3 src/Digraph.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --alpha='+str(alpha)
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' --epochs='+epochs
                            +' --method_name=DiGib')
                print(command)
                os.system(command)
    