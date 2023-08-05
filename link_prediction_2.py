import os, sys

epochs = '3000'
for data in [#'bitcoin_alpha', 'bitcoin_otc', 
     #'telegram/telegram' 
#'dataset_nodes500_alpha0.05_beta0.2', 'dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2'
#'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue',
#'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue',
#'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue'
#'dataset_nodes500_alpha0.05_beta0.2', 'dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2'
'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue',
'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue',
'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue'
]:
    for task in [
         'three_class_digraph'
         #'direction', 
         #'existence'#,'all'
                ]:
        num_class_link = 2
        if task == 'three_class_digraph':
                num_class_link = 3
        for lr in [1e-3]:
            log_path = 'Edge_'+data[:-1]+'_Digraph'
            for num_filter in [#6,11, 
            21
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
                                +' --noisy'
                                +' --method_name=DiGib')
                    print(command)
                    os.system(command)
            
        ##
            log_path = 'Edge_'+data[:-1]+'_APPNP'
            for num_filter in [#16, 32, 
                 48
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
            
            log_path = 'Edge_'+data[:-1]+'_DiGCL'
            for num_filter in [64]:
                for curr_type in ["linear", "exp", "log", "fixed"
                ]:
                    command = ('python3 src/Edge_DiGCL.py ' 
                            +' --dataset='+data
                            +' --task='+ task
                            +' --num_class_link='+str(num_class_link)
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --lr='+str(lr)
                            +' --epochs='+epochs
                            +' --activation=relu'                            
                            +' --noisy'
                            +' --curr-type='+str(curr_type)
                            +' --method_name=DiGCL'
                            +' --weight_decay=0.0005'
                            +' --drop_feature_rate_1=0.3'
                            +' --drop_feature_rate_2=0.4'
                            + ' --tau=0.4')
                    print(command)
                    os.system(command)

                    command = ('python3 src/Edge_DiGCL.py ' 
                            +' --dataset='+data
                            +' --task='+ task
                            +' --num_class_link='+str(num_class_link)
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --lr='+str(lr)
                            +' --epochs='+epochs
                            +' --activation=prelu'                            
                            +' --noisy'
                            +' --curr-type='+str(curr_type)
                            +' --method_name=DiGCL'
                            +' --weight_decay=0.0005'
                            +' --drop_feature_rate_1=0.2'
                            +' --drop_feature_rate_2=0.1'
                            +' --tau=0.9')
                    print(command)
                    os.system(command)