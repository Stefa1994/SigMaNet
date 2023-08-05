import numpy as np
import torch
import pickle as pk
import torch.optim as optim
from datetime import datetime
import os, time, argparse
import torch.nn.functional as F
from torch_geometric_signed_directed.data import load_directed_real_data
from torch_geometric_signed_directed.data.signed import load_signed_real_data
import random

# internal files
from layer.Signum import SigMaNet_link_prediction_one_laplacian
from layer.src2 import laplacian
from utils.edge_data import link_class_split, in_out_degree, load_signed_real_data_no_negative, load_signed_real_data_also_negative
from utils.save_settings import write_log
from utils.edge_data_new import link_class_split_new

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="link prediction of SigNum")
    parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='WebKB/Cornell', help='data set selection')
    
    
    parser.add_argument('--split_prob', type=lambda s: [float(item) for item in s.split(',')], default="0.05,0.15", help='random drop for testing/validation/training edges (for 3-class classification only)')
    parser.add_argument('--task', type=str, default='direction', help='Task')
    
    parser.add_argument('--epochs', type=int, default=1500, help='training epochs')
    parser.add_argument('--num_filter', type=int, default=4, help='num of filters')
    parser.add_argument('--method_name', type=str, default='SigNum', help='method name')

    parser.add_argument('--K', type=int, default=1, help='K for cheb series')
    parser.add_argument('--layer', type=int, default=2, help='how many layers of gcn in the model, only 1 or 2 layers.')
    parser.add_argument('--netflow', '-N', action='store_false', help='if use net flow')
    parser.add_argument('--follow_math', '-F', action='store_false', help='if follow math')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--debug', '-D', action='store_true', help='debug mode')
    parser.add_argument('--num_class_link', type=int, default=2,
                        help='number of classes for link direction prediction(2 or 3).')

    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')
    parser.add_argument('--noisy',  action='store_true')
    parser.add_argument('--randomseed', type=int, default=0, help='if set random seed in training')


    return parser.parse_args()

# Inserire il netflow come argomento esterno

def acc(pred, label):
    #print(pred.shape, label.shape)       
    correct = pred.eq(label).sum().item()
    acc = correct / len(pred)
    return acc

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.cofollow_mathl)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def main(args):

    random.seed(args.randomseed)
    torch.manual_seed(args.randomseed)
    np.random.seed(args.randomseed)

    #follow_math = False
    gcn = True
    #net_flow = True

    date_time = datetime.now().strftime('%m-%d-%H:%M:%S')
    log_path = os.path.join(args.log_root, args.log_path, args.save_name, date_time)

    
    if os.path.isdir(log_path) == False:
        os.makedirs(log_path)
    
    
    dataset_name = args.dataset.split('/')
    if args.dataset in ['telegram']:
        data = load_directed_real_data(dataset=dataset_name[0], name=dataset_name[0]).to(device)
        data = data.to(device)
        subset = args.dataset
    else:
        #data = load_signed_real_data_no_negative(dataset=args.dataset).to(device)
        #data, edge_neg, weight_neg = load_signed_real_data_also_negative(dataset=args.dataset)
        if args.dataset in ['bitcoin_alpha', 'bitcoin_otc']:
            #data = load_signed_real_data(dataset=args.dataset).to(device)
            data = load_signed_real_data_no_negative(dataset=args.dataset).to(device)
        else:
            try:
                data = pk.load(open(f'./data/fake/{args.dataset}.pk','rb'))
            except:
                data = pk.load(open(f'./data/fake_for_quaternion_new/{args.dataset}.pk','rb'))
            data = data.to(device)
        #subset = args.dataset        
        #data = data.to(device)
        subset = args.dataset
    edge_index = data.edge_index
        
    # load dataset
    #if 'dataset' in locals():
    #    data = dataset[0]
    #    edge_index = data.edge_index
        #feature = dataset[0].x.data

    size = torch.max(edge_index).item()+1
    data.num_nodes = size
    # generate edge index dataset
    #if args.task == 2:
    #    datasets = generate_dataset_2class(edge_index, splits = 10, test_prob = args.drop_prob)
    #else:
    save_file = args.data_path + args.dataset + '/' + subset
    #datasets = link_class_split(data, prob_val=args.split_prob[0], prob_test=args.split_prob[1], splits = 10, task = args.task, noisy = args.noisy)
    datasets = link_class_split_new(data, prob_val=args.split_prob[0], prob_test=args.split_prob[1], splits = 10, task = args.task)


    #if args.task != 'direction':
    also_neg = False
    results = np.zeros((10, 4))
    #else:
    #    results = np.zeros((10, 4, 5))
    for i in range(10):
        log_str_full = ''

        ########################################
        # get hermitian laplacian
        ########################################
        edge_index = datasets[i]['graph']
        edge_weight = datasets[i]['weights']
        #if also_neg:
        #    edge_index = torch.cat((edge_index, edge_neg), axis = 1)
        #    edge_weight = torch.cat((edge_weight, weight_neg), axis = 0)
        #    size = torch.max(edge_index).item()+1
        X_real = in_out_degree(edge_index, size, edge_weight).to(device)
        #X_real = in_out_degree(edges, size).to(device)
        X_img = X_real.clone()
        #X_img = torch.zeros(X_real.size()).to( device=X_real.device)
        #exit()
        #else:
        #    X_img  = torch.ones(L_real[0].shape[-1]).unsqueeze(-1).to(device)
        #    X_real = torch.ones(L_real[0].shape[-1]).unsqueeze(-1).to(device)

        ########################################
        # initialize model and load dataset
        ########################################
        edge_index, norm_real, norm_imag = laplacian.process_magnetic_laplacian(edge_index=edge_index, gcn=gcn, net_flow=args.netflow, x_real=X_real, edge_weight=edge_weight, \
         normalization = 'sym', return_lambda_max = False)
        model = SigMaNet_link_prediction_one_laplacian(K=args.K, num_features=2, hidden=args.num_filter, label_dim=args.num_class_link,
                            i_complex = False,  layer=args.layer, follow_math=args.follow_math, gcn =gcn, net_flow=args.netflow, unwind = True, edge_index=edge_index,\
                            norm_real=norm_real, norm_imag=norm_imag).to(device)

        #model = nn.DataParallel(model)  
        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        y_train = datasets[i]['train']['label']
        y_val   = datasets[i]['val']['label']
        y_test  = datasets[i]['test']['label']
        y_train = y_train.long().to(device)
        y_val   = y_val.long().to(device)
        y_test  = y_test.long().to(device)

        train_index = datasets[i]['train']['edges'].to(device)
        val_index = datasets[i]['val']['edges'].to(device)
        test_index = datasets[i]['test']['edges'].to(device)
        
        #################################
        # Train/Validation/Test
        #################################
        best_test_err = 1000.0
        best_test_acc = 0.0
        early_stopping = 0
        for epoch in range(args.epochs):
            start_time = time.time()
            if early_stopping > 500:
                break
            ####################
            # Train
            ####################
            train_loss, train_acc = 0.0, 0.0
            model.train()
        
            out = model(X_real, X_img, train_index)

            train_loss = F.nll_loss(out, y_train)
            pred_label = out.max(dim = 1)[1]            
            train_acc  = acc(pred_label, y_train)
            
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            outstrtrain = 'Train loss: %.6f, acc: %.3f' % (train_loss.detach().item(), train_acc)
            
            ####################
            # Validation
            ####################
            train_loss, train_acc = 0.0, 0.0
            model.eval()
            out = model(X_real, X_img, 
                        val_index)

            test_loss  = F.nll_loss(out, y_val)
            pred_label = out.max(dim = 1)[1]            
            test_acc   = acc(pred_label, y_val)

            outstrval = ' Test loss: %.6f, acc: %.3f' % (test_loss.detach().item(), test_acc)            
            duration = "--- %.4f seconds ---" % (time.time() - start_time)
            log_str = ("%d / %d epoch" % (epoch, args.epochs))+outstrtrain+outstrval+duration
            #print(log_str)
            log_str_full += log_str + '\n'
            ####################
            # Save weights
            ####################
            save_perform_err = test_loss.detach().item()
            save_perform_acc = test_acc
            if save_perform_err <= best_test_err:
                early_stopping = 0
                best_test_err = save_perform_err
                torch.save(model.state_dict(), log_path + '/model_err'+str(i)+'.t7')
            if save_perform_acc >= best_test_acc:
                #early_stopping = 0
                best_test_acc = save_perform_acc
                torch.save(model.state_dict(), log_path + '/model_acc'+str(i)+'.t7')
            else:
                early_stopping += 1
        torch.save(model.state_dict(), log_path + '/model_latest'+str(i)+'.t7')
        write_log(vars(args), log_path)

        #if args.task == 'existence':
            ####################
            # Testing
            ####################
        model.load_state_dict(torch.load(log_path + '/model_err'+str(i)+'.t7'))
        model.eval()
        out = model(X_real, X_img, val_index)
        pred_label = out.max(dim = 1)[1]
        val_err = acc(pred_label, y_val)
        out = model(X_real, X_img, test_index)
        pred_label = out.max(dim = 1)[1]
        test_err = acc(pred_label, y_test)

        model.load_state_dict(torch.load(log_path + '/model_acc'+str(i)+'.t7'))
        model.eval()
        out = model(X_real, X_img, val_index)
        pred_label = out.max(dim = 1)[1]
        val_acc_err = acc(pred_label, y_val)
        out = model(X_real, X_img, test_index)
        pred_label = out.max(dim = 1)[1]
        test_acc_err = acc(pred_label, y_test)
        #print('loss', test_err)
        #print('accuracy', test_acc_err)
        #if test_err >= test_acc_err:
        test_acc = test_err
        val_acc = val_err
        #else:
        #    test_acc = test_acc_err
        #    val_acc = val_acc_err

   
        model.load_state_dict(torch.load(log_path + '/model_latest'+str(i)+'.t7'))
        model.eval()
        out = model(X_real, X_img, val_index)
        pred_label = out.max(dim = 1)[1]
        val_acc_latest = acc(pred_label, y_val)
    
        out = model(X_real, X_img, test_index)
        pred_label = out.max(dim = 1)[1]
        test_acc_latest = acc(pred_label, y_test)
        ####################
        # Save testing results
        ####################
        log_str = ('val_acc: {val_acc:.4f}, '+'test_acc: {test_acc:.4f}, ')
        log_str1 = log_str.format(val_acc = val_acc, test_acc = test_acc)
        log_str_full += log_str1
        log_str = ('val_acc_latest: {val_acc_latest:.4f}, ' + 'test_acc_latest: {test_acc_latest:.4f}, ' )
        log_str2 = log_str.format(val_acc_latest = val_acc_latest, test_acc_latest = test_acc_latest)
        log_str_full += log_str2 + '\n'
        print(log_str1+log_str2)
        results[i] = [val_acc, test_acc, val_acc_latest, test_acc_latest]
        #else:
        #    model.load_state_dict(torch.load(log_path + '/model'+str(i)+'.t7'))
        #    model.eval()
        #    out_val = model(X_real, X_img, val_index)
        #    out_test = model(X_real, X_img, test_index)
        #    [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro],
        #        [test_acc_full, test_acc, test_auc, 
        #        test_f1_micro, test_f1_macro]] = link_prediction_evaluation(out_val, out_test, y_val, y_test)
            
        #    model.load_state_dict(torch.load(log_path + '/model_latest'+str(i)+'.t7'))
        #    model.eval()
        #    out_val = model(X_real, X_img, val_index)
        #    out_test = model(X_real, X_img, test_index)
        #    [[val_acc_full_latest, val_acc_latest, val_auc_latest, val_f1_micro_latest, val_f1_macro_latest],
        #                    [test_acc_full_latest, test_acc_latest, test_auc_latest, 
        #                    test_f1_micro_latest, test_f1_macro_latest]] = link_prediction_evaluation(out_val, out_test, y_val, y_test)
            ####################
            # Save testing results
            ####################
        #    log_str = ('val_acc_full:{val_acc_full:.4f}, val_acc: {val_acc:.4f}, Val_auc: {val_auc:.4f},'
        #                + 'val_f1_micro: {val_f1_micro:.4f}, val_f1_macro: {val_f1_macro:.4f}, ' 
        #                + 'test_acc_full:{test_acc_full:.4f}, test_acc: {test_acc:.4f}, '
        #                + 'test_f1_micro: {test_f1_micro:.4f}, test_f1_macro: {test_f1_macro:.4f}')
        #    log_str = log_str.format(val_acc_full = val_acc_full, 
        #                                val_acc = val_acc, val_auc = val_auc, val_f1_micro = val_f1_micro, 
        #                                val_f1_macro = val_f1_macro, test_acc_full = test_acc_full, 
        #                                test_acc = val_acc, 
        #                                test_f1_micro = val_f1_micro, test_f1_macro = val_f1_macro)
        #    log_str_full += log_str + '\n'
        #    print(log_str)

        #    log_str = ('val_acc_full_latest:{val_acc_full_latest:.4f}, val_acc_latest: {val_acc_latest:.4f}, Val_auc_latest: {val_auc_latest:.4f},' 
        #                + 'val_f1_micro_latest: {val_f1_micro_latest:.4f}, val_f1_macro_latest: {val_f1_macro_latest:.4f},' 
        #                + 'test_acc_full_latest:{test_acc_full_latest:.4f}, test_acc_latest: {test_acc_latest:.4f}, ' 
        #                + 'test_f1_micro_latest: {test_f1_micro_latest:.4f}, test_f1_macro_latest: {test_f1_macro_latest:.4f}')
        #    log_str = log_str.format(val_acc_full_latest = val_acc_full_latest, 
        #    val_acc_latest = val_acc_latest, val_auc_latest = val_auc_latest,
        #                            val_f1_micro_latest = test_f1_micro_latest, val_f1_macro_latest = val_f1_macro_latest,
        #                            test_acc_full_latest = test_acc_full_latest,
        #                            test_acc_latest = val_acc, test_f1_micro_latest = test_f1_micro_latest, test_f1_macro_latest = test_f1_macro_latest)
        #    log_str_full += log_str + '\n'
        #    print(log_str)

        #    results[i] = [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro],
        #                    [test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro],
        #                    [val_acc_full_latest, val_acc_latest, val_auc_latest, val_f1_micro_latest, val_f1_macro_latest],
        #                    [test_acc_full_latest, test_acc_latest, test_auc_latest, test_f1_micro_latest, test_f1_macro_latest]]
        with open(log_path + '/log'+str(i)+'.csv', 'w') as file:
            file.write(log_str_full)
            file.write('\n')
        torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.epochs = 1

    save_name = args.method_name + 'lr' + str(int(args.lr*1000)) + 'num_filters' + str(int(args.num_filter)) + 'task' + args.task + 'layers' + str(args.layer) + 'net_flow' + str(args.netflow)  + '_noisy' +  str(args.noisy) + '_math' + str(args.follow_math)
    args.save_name = save_name

    args.log_path = os.path.join(args.log_path,args.method_name, args.dataset)
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), '../result_arrays',args.log_path,args.dataset+'/')
    if os.path.isdir(dir_name) == False:
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            print('Folder exists!')

    results = main(args)
    np.save(dir_name+save_name, results)