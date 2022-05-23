import os, sys

epochs = '3000'
for data in ['bitcoin_otc', 'bitcoin_alpha', 'telegram' ]:
    for task in ['direction', 'existence','all'
                ]:
        num_class_link = 2
        if task == 'all':
                num_class_link = 3
        for lr in [1e-3]:
            log_path = 'Edge_'+data[:-1]
            for num_filter in [16 , 32, 64]:
                for q in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25
                                ]:
                    command = ('python3 src/Edge_sparseMagnet.py ' 
                                +' --dataset='+data
                                +' --q='+str(q)
                                +' --num_filter='+str(num_filter)
                                +' --K=1'
                                #+' -D'
                                +' --num_class_link='+str(num_class_link)
                                +' --log_path='+str(log_path)
                                +' --layer=2'
                                +' --epochs='+epochs
                                +' --lr='+str(lr)
                                +' --task='+ task
                                +' -a'
                                +' --noisy')
#
                    print(command)
                    os.system(command)
#
            log_path = 'Edge_'+data[:-1]+'_SymDiGCN'
            for num_filter in [5, 15, 30
            ]:
                command = ('python3 src/Edge_SymDiGCN.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --lr='+str(lr)
                            +' --task='+ task
                            +' --num_class_link='+str(num_class_link)
                            #+' -D'
                            +' --epochs='+epochs
                            +' --noisy')
                print(command)
                os.system(command)
##
            log_path = 'Edge_'+data[:-1]+'_Cheb'
            for num_filter in [16, 32, 64
            ]:
                    command = ('python3 src/Edge_Cheb.py ' 
                                +' --dataset='+data
                                +' --K=2'
                                #+' -D'
                                +' --task='+ task
                                +' --num_filter='+str(num_filter)
                                +' --log_path='+str(log_path)
                                +' --num_class_link='+str(num_class_link)
                                +' --lr='+str(lr)
                                +' --epochs='+epochs
                                +' --noisy')
                    print(command)
                    os.system(command)
##
            log_path = 'Edge_'+data[:-1]+'_GCN'
            for num_filter in [16, 32, 64
            ]:
                    command = ('python3 src/Edge_GCN.py ' 
                                +' --dataset='+data
                                #+' -D'
                                +' --task='+ task
                                +' --num_filter='+str(num_filter)
                                +' --log_path='+str(log_path)
                                +' --num_class_link='+str(num_class_link)
                                +' --epochs='+epochs
                                +' --lr='+str(lr)
                                +' -tud'
                                +' --noisy')
                    print(command)
                    os.system(command)
#
            log_path = 'Edge_'+data[:-1]+'_SAGE'
            for num_filter in [16, 32, 64
            ]:
                    command = ('python3 src/Edge_SAGE.py ' 
                                +' --dataset='+data
                                +' --task='+ task
                                +' --num_filter='+str(num_filter)
                                +' --num_class_link='+str(num_class_link)
                                +' --log_path='+str(log_path)
                                +' --lr='+str(lr)
                                +' --epochs='+epochs
                                +' --noisy')
                    print(command)
                    os.system(command)
                    #command = ('python3 src/Edge_SAGE.py ' 
                    #        +' --dataset='+data
                    #        +' --task='+ task
                    #        +' --num_filter='+str(num_filter)
                    #        +' --num_class_link='+str(num_class_link)
                    #        +' --log_path='+str(log_path)
                    #        +' --epochs='+epochs
                    #        +' --lr='+str(lr)
                    #        +' -tud'
                    #        +' --noisy')
                    #print(command)
                    #os.system(command)
#
            log_path = 'Edge_'+data[:-1]+'_GAT'
            for heads in [2, 4, 8
            ]:
                for num_filter in [16, 32, 64
                ]:
                    command = ('python3 src/Edge_GAT.py ' 
                                +' --dataset='+data
                                +' --task='+ task
                                +' --heads='+str(heads)
                                +' --num_class_link='+str(num_class_link)
                                +' --num_filter='+str(num_filter)
                                +' --log_path='+str(log_path)
                                +' --lr='+str(lr)
                                +' --epochs='+epochs
                                +' --noisy')
                    print(command)
                    os.system(command) 
                    #command = ('python3 src/Edge_GAT.py ' 
                    #            +' --dataset='+data
                    #            +' --task='+ task
                    #            +' --heads='+str(heads)
                    #            +' --num_filter='+str(num_filter)
                    #            +' --num_class_link='+str(num_class_link)
                    #            +' --log_path='+str(log_path)
                    #            +' --lr='+str(lr)
                    #            +' --epochs='+epochs
                    #            +' -tud'
                    #            +' --noisy')
                    #print(command)
                    #os.system(command) 
#
#
            log_path = 'Edge_'+data[:-1]+'_GIN'
            for num_filter in [16, 32, 64
            ]:
                    command = ('python3 src/Edge_GIN.py ' 
                                +' --dataset='+data
                                #+' -D'
                                +' --task='+ task
                                +' --num_filter='+str(num_filter)
                                +' --num_class_link='+str(num_class_link)
                                +' --log_path='+str(log_path)
                                +' --lr='+str(lr)
                                +' --epochs='+epochs
                                +' --noisy')
                    print(command)
                    os.system(command)
        ##
            log_path = 'Edge_'+data[:-1]+'_APPNP'
            for num_filter in [16, 32, 48
            ]:
                for alpha in [0.05, 0.1, 0.15, 0.2
                ]: 
                    for K in [1, 5, 10
                    ]: 
                        command = ('python3 src/Edge_APPNP.py ' 
                                    +' --dataset='+data
            #                        +' -D'
                                    +' --task='+ task
                                    +' --num_filter='+str(num_filter)
                                    +' --num_class_link='+str(num_class_link)
                                    +' --log_path='+str(log_path)
                                    +' --lr='+str(lr)
                                    +' --K='+str(K)
                                    +' --alpha='+str(alpha)
                                    +' --epochs='+epochs
                                    +' --noisy')
                        print(command)
                        os.system(command)
            #            command = ('python3 Edge_APPNP.py ' 
            ##                        +' --dataset='+data
            ##                        +' -D'
            ##                        +' --task='+ task
            ##                        +' --num_filter='+str(num_filter)
            ##                        +' --num_class_link='+str(num_class_link)
            ##                        +' --log_path='+str(log_path)
            ##                        +' --epochs='+epochs
            ##                        +' --lr='+str(lr)
            ##                        +' --K='+str(K)
            ##                        +' --alpha='+str(alpha)
            ##                        +' -tud')
            ##            print(command)
            ##            os.system(command)
        ##
            log_path = 'Edge_'+data[:-1]+'_Digraph'
            for num_filter in [16, 32, 64
            ]:
                for alpha in [0.1, 0.05, 0.15, 0.2
                ]: 
                    command = ('python3 src/Edge_Digraph.py ' 
                                +' --dataset='+data
            #                    +' -D'
                                +' --task='+ task
                                +' --num_filter='+str(num_filter)
                                +' --num_class_link='+str(num_class_link)
                                +' --log_path='+str(log_path)
                                +' --lr='+str(lr)
                                +' --alpha='+str(alpha)
                                +' --epochs='+epochs
                                +' --noisy')
                    print(command)
                    os.system(command)