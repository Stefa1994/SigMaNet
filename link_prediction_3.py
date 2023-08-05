import os, sys


epochs = '3000'
for data in [ #'bitcoin_otc', 
     #'bitcoin_alpha', 'telegram'
        #'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.4_opposite-signFalse_negative-edgesFalse_directedTrue.pk'
#'dataset_nodes150_alpha0.6_beta0.1_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue.pk',
#'dataset_nodes150_alpha0.6_beta0.1_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue.pk',
#'dataset_nodes150_alpha0.6_beta0.1_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue.pk',
#'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue',
#'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue',
#'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue'
#'dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2', 'dataset_nodes500_alpha0.05_beta0.2'
'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue',
'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue',
'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue'
             ]:
    for task in [
         'three_class_digraph'
         #'direction', 
         #'existence', #'all'
    ]:
        num_class_link = 2
        if task == 'three_class_digraph':
                num_class_link = 3
        for layer in [2]:
            for lr in [1e-3]:
                log_path = 'Edge_trick_'+data+'_SigMaNet'
                for num_filter in [ #16, 32,  
                64]:
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
                log_path = 'Edge_'+data[:-1]+'_GAT'
                for heads in [2, 4, 8
                ]:
                        for num_filter in [64
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

                log_path = 'Edge_'+data+'_QGNN'
                for num_filter in [ #16, 32,  
                64]:
                        command = ('python3 src/Edge_QGNN.py ' 
                                    +' --dataset='+data
                                    +' --num_filter='+str(num_filter)
                                    #+' -D'
                                    +' --num_class_link='+str(num_class_link)
                                    +' --log_path='+str(log_path)
                                    +' --epochs='+epochs
                                    +' --lr='+str(lr)
                                    +' --task='+ task
                                    +' --noisy')
                        print(command)
                        os.system(command)

                
                log_path = 'Edge_'+data[:-1]+'_Digraph'
                for num_filter in [#16, 32, 
                 64
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
                log_path = 'Edge_'+data[:-1]+'_GIN'
                for num_filter in [#16, 32, 
                64]:
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