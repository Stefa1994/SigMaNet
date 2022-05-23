import os, sys


epochs = '3000'
for data in ['bitcoin_otc', 'bitcoin_alpha', 'telegram' ]:
    for task in ['direction', 'existence', 'all']:
        num_class_link = 2
        if task == 'all':
                num_class_link = 3
        for layer in [2]:
            for lr in [1e-3]:
                log_path = 'Edge'+data+'_SigMaNet'
                for num_filter in [ 16, 32,  64]:
                        command = ('python3 src/Edge_SigMaNet.py ' 
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