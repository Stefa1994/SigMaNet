import os, sys


epochs = '3000'
for data in ['dataset_nodes500_alpha0.05_beta0.2', 'dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2']:
    for task in ['direction', 'existence']: 
        num_class_link = 2
        for layer in [2]:
            for lr in [1e-3]:
                log_path = 'Edge'+data+'_SigMaNet'
                for num_filter in [ 16, 32,  64]:
                        command = ('python3 Edge_SigMaNet.py ' 
                                    +' --dataset='+data
                                    +' --num_filter='+str(num_filter)
                                    +' --K=1'
                                    #+' -D'
                                    +' --num_class_link='+str(num_class_link)
                                    +' --log_path='+str(log_path)
                                    +' --layer='+str(layer)
                                    +' --epochs='+epochs
                                    +' --lr='+str(lr)
                                    +' --task='+ task
                                    +' -N'
                                    +' -F'
                                    +' --noisy')
                        print(command)
                        os.system(command)