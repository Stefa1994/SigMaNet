import os, sys


epochs = '3000'
for data in [#'dataset_nodes500_alpha1_beta0.95_undirected-percentage0.8_opposite-signFalse_negative-edgesFalse_directedTrue'] 
    #'bitcoin_otc', 'bitcoin_alpha', 
    #'telegram'
#            'dataset_nodes500_alpha0.05_beta0.2', 'dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2'
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
    
#dir = '/home/gpu2/Documenti/SigMaNet/data/fake_for_quaternion_new'
#for data in os.listdir(dir):    
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
                log_path = 'Edge_'+data[:-3]+'_QuaNet_Data_Test'
                for num_filter in [64]:
                        #command = ('python3 src/Edge_QuaNet.py ' 
                        #            +' --dataset='+data
                        #            +' --num_filter='+str(num_filter)
                        #            +' --K=1'
                        #            #+' -D'
                        #            +' --num_class_link='+str(num_class_link)
                        #            +' --log_path='+str(log_path)
                        #            +' --layer='+str(layer)
                        #            +' --epochs='+epochs
                        #            +' --lr='+str(lr)
                        #            +' --task='+ task
                        #            +' --noisy')
                        #print(command)
                        #os.system(command)

                        command = ('python3 src/Edge_QuaNet.py ' 
                                    +' --dataset='+data
                                    +' --num_filter='+str(num_filter)
                                    +' --K=1'
                                    +' -W'
                                    +' -B'
                                    +' --num_class_link='+str(num_class_link)
                                    +' --log_path='+str(log_path)
                                    +' --layer='+str(layer)
                                    +' --epochs='+epochs
                                    +' --lr='+str(lr)
                                    +' --task='+ task
                                    +' --noisy')
                        print(command)
                        os.system(command)