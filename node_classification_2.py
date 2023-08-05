import os, sys

epochs = '3000'
for data in [#'telegram',  
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue',
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue',
    'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue'
    #'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue.pk',
    #'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue.pk',
    #'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue.pk'
]:
    for lr in [1e-3]:
        log_path = 'DiGCL_' + data
        for num_filter in [64]:
            for curr_type in ["linear", "exp", "log", "fixed"
            ]:
                command = ('python3 src/DiGCL.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --activation=relu'                            
                            +' --lr='+str(lr)
                            +' --epochs='+epochs
                            +' --curr-type='+str(curr_type)
                            +' --weight_decay=0.0005'
                            +' --drop_feature_rate_1=0.3'
                            +' --drop_feature_rate_2=0.4'
                            + ' --tau=0.4')
                print(command)
                os.system(command)

                command = ('python3 src/DiGCL.py ' 
                            +' --dataset='+data
                            +' --num_filter='+str(num_filter)
                            +' --log_path='+str(log_path)
                            +' --activation=prelu'                            
                            +' --lr='+str(lr)
                            +' --epochs='+epochs
                            +' --curr-type='+str(curr_type)
                            +' --weight_decay=0.0005'
                            +' --drop_feature_rate_1=0.2'
                            +' --drop_feature_rate_2=0.1'
                            +' --tau=0.9')
                print(command)
                os.system(command)

        log_path = 'DiGraph_' + data
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

        log_path = 'QGNN_' + data
        for num_filter in [64]:
            command = ('python3 src/QGNN.py ' 
                        +' --dataset='+data
                        +' --num_filter='+str(num_filter)
                        +' --log_path='+str(log_path)
                        +' --dropout=0.5'
                        +' --lr='+str(lr)
                        +' --epochs='+epochs)
            print(command)
            os.system(command)


            
    