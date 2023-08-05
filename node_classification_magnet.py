import os

epochs = '3000'
for data in [#'dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2', 'dataset_nodes500_alpha0.05_beta0.2'
            'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.8_opposite-signFalse_negative-edgesFalse_directedTrue.pk'
            ]:
    for lr in [1e-3]:#, 1e-2, 5e-3]:
        # MagNet
        log_path = 'Sym_' + data
        for num_filter in [60]:
            for q in [0.01, 0.05, 0.1, 0.15, 0.2, 
            0.25]:
                command = ('python3 src/sparse_Magnet.py ' 
                            +' --dataset='+data
                            +' --q='+str(q)
                            +' --num_filter='+str(num_filter)
                            +' --K=1'
                            +' --log_path='+str(log_path)
                            +' --layer=2'
                            +' --epochs='+epochs
                            +' --dropout=0.5'
                            +' --lr='+str(lr)
                            +' -a')
                print(command)
                os.system(command)