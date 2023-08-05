import os, sys

epochs = '3000'
for data in [ #'dataset_nodes100_alpha0.1_beta0.2_undirected-percentage0.1_opposite-signFalse_negative-edgesFalse_directedTrue'
     
     #'telegram/telegram' #,  
     #'dataset_nodes500_alpha0.05_beta0.2', 'dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2'
    #'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue.pk',
    #'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue.pk',
    #'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue.pk'
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue',
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue',
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue'

]:
#dir = '/home/gpu2/Documenti/SigMaNet/data/fake_for_quaternion_new'

#for data in os.listdir(dir):
    for lr in [1e-3]:
        # QuaNet
        log_path = 'QuaNet_' + data
        for num_filter in [#16, 32,  
        64]:

                command = ('python3 src/QuaNet.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --K=1'
                            +' --log_path='+str(log_path)
                            +' --layer=2'
                            +' --epochs='+epochs
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' -W'
                            +' -B')
                print(command)
                os.system(command)
        