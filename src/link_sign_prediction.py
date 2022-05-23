import os, sys

epochs = '3000'
for data in [ 'bitcoin_alpha', 'bitcoin_otc', 'edges', 'slashdot',  'epinions' ]:
    for task in ['sign']: 
        for layer in [2]:
        #3, 4]: 
            for lr in [1e-2, 1e-3, 5e-3]:
                log_path = 'Edge_sign_'+data+'_SigNum'
                for num_filter in [16, 32, 64]:
                        command = ('python3 Edge_SigNum_sign.py ' 
                                    +' --dataset='+data
                                    +' --num_filter='+str(num_filter)
                                    +' --K=1'
                                    #+' -D'
                                    #+' --num_class_link='+str(num_class_link)
                                    +' --log_path='+str(log_path)
                                    +' --layer='+str(layer)
                                    +' --epochs='+epochs
                                    +' --lr='+str(lr)
                                    +' --task='+ task
                                    +' -N'
                                    +' -F')
                        print(command)
                        os.system(command)
                        command = ('python3 Edge_SigNum_sign.py ' 
                                    +' --dataset='+data
                                    +' --num_filter='+str(num_filter)
                                    +' --K=1'
                                    #+' -D'
                                    #+' --num_class_link='+str(num_class_link)
                                    +' --log_path='+str(log_path)
                                    +' --layer='+str(layer)
                                    +' --epochs='+epochs
                                    +' --lr='+str(lr)
                                    +' --task='+ task
                                    +' -F')
                        print(command)
                        os.system(command)