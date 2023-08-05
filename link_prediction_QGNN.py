import os, sys


epochs = '3000'
for data in [ #'bitcoin_otc', 'bitcoin_alpha', 'telegram/telegram' 
    #'dataset_nodes150_alpha0.6_beta0.1_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue.pk',
#'dataset_nodes150_alpha0.6_beta0.1_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue.pk',
#'dataset_nodes150_alpha0.6_beta0.1_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue.pk',
#'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue',
#'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue',
#'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue'
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