# Controllare se posso fare input weight

import numpy as np
import pickle as pk
import torch.optim as optim
from datetime import datetime
import os, time, argparse
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_geometric_signed_directed.data import load_directed_real_data
import random
import pickle as pk



# internal files
from layer.geometric_baselines import *
from utils.edge_data import link_class_split, in_out_degree,  load_signed_real_data_no_negative
from utils.save_settings import write_log
from utils.edge_data_new import link_class_split_new


# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="link prediction baseline--GAT")

    parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='WebKB/Cornell', help='data set selection')
    
    
    parser.add_argument('--split_prob', type=lambda s: [float(item) for item in s.split(',')], default="0.05,0.15", help='random drop for testing/validation/training edges (for 3-class classification only)')
    parser.add_argument('--task', type=str, default='direction', help='Task')

    parser.add_argument('-hds', '--heads', default=2, type=int)
    parser.add_argument('--method_name', type=str, default='GAT', help='method name')
    parser.add_argument('--debug', '-D', action='store_true', help='debug mode')
    parser.add_argument('--num_class_link', type=int, default=2,
                        help='number of classes for link direction prediction(2 or 3).')


    parser.add_argument('--epochs', type=int, default=1500, help='training epochs')
    parser.add_argument('--num_filter', type=int, default=4, help='num of filters')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('-to_undirected', '-tud', action='store_true', help='if convert graph to undirecteds')
    
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')
    parser.add_argument('--noisy',  action='store_true')
    parser.add_argument('--randomseed', type=int, default=0, help='if set random seed in training')


    return parser.parse_args()

def acc(pred, label):
    correct = pred.eq(label).sum().item()
    acc = correct / len(pred)
    return acc

def main(args):

    random.seed(args.randomseed)
    torch.manual_seed(args.randomseed)
    np.random.seed(args.randomseed)


    date_time = datetime.now().strftime('%m-%d-%H:%M:%S')
    log_path = os.path.join(args.log_root, args.log_path, args.save_name, date_time)

  

    if os.path.isdir(log_path) == False:
        os.makedirs(log_path)
    
    
    dataset_name = args.dataset.split('/')
    if len(dataset_name) == 1:
        if args.dataset in ['bitcoin_alpha', 'bitcoin_otc']:
            data = load_signed_real_data_no_negative(dataset=args.dataset).to(device)
        else:
            try:
                data = pk.load(open(f'./data/fake/{args.dataset}.pk','rb'))
            except:
                data = pk.load(open(f'./data/fake_for_quaternion_new/{args.dataset}.pk','rb'))
            data = data.to(device)
        subset = args.dataset
    else:
        load_func, subset = args.dataset.split('/')[0], args.dataset.split('/')[1]
     #save_name = args.method_name + '_' + 'Layer' + str(args.layer) + '_' + 'lr' + str(args.lr) + 'num_filters' + str(int(args.num_filter))+ '_' + 'task' + str((args.task))
        data = load_directed_real_data(dataset=dataset_name[0], name=dataset_name[1]).to(device)
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

    #if args.task == 'existence':
    results = np.zeros((10, 4))
    #else:
    #    results = np.zeros((10, 4, 5))
    for i in range(10):
        log_str_full = ''
        edges = datasets[i]['graph']
        edge_weight = datasets[i]['weights']
        edge_weight = torch.FloatTensor(edge_weight)
        if args.to_undirected:
            edges, edge_weight = to_undirected(edges, edge_weight)
        
        ########################################
        # initialize model and load dataset
        ########################################
        #x = torch.ones(size).unsqueeze(-1).to(device)
        x = in_out_degree(edges, size,  edge_weight).to(device)
        edges = edges.long().to(device)

        model = GAT_Link(x.size(-1), args.num_class_link, heads = args.heads, filter_num=args.num_filter, dropout=args.dropout).to(device)
        #model = nn.DataParallel(graphmodel)
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
        best_test_err = 1000000.0
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
            out = model(x, edges, train_index)

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
            out = model(x, edges, val_index)

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
            torch.save(model.state_dict(), log_path + '/model_latest'+str(i)+'.t7')
            save_perform = test_loss.detach().item()
            if save_perform <= best_test_err:
                early_stopping = 0
                best_test_err = save_perform
                torch.save(model.state_dict(), log_path + '/model'+str(i)+'.t7')
            else:
                early_stopping += 1

        write_log(vars(args), log_path)
        torch.save(model.state_dict(), log_path + '/model_latest'+str(i)+'.t7')
        #if args.task == 'existence':
            ####################
            # Testing
            ####################
        model.load_state_dict(torch.load(log_path + '/model'+str(i)+'.t7'))
        model.eval()
        out = model(x, edges, val_index)
        pred_label = out.max(dim = 1)[1]
        val_acc = acc(pred_label, y_val)
        out = model(x, edges, test_index)
        pred_label = out.max(dim = 1)[1]
        test_acc = acc(pred_label, y_test)
        model.load_state_dict(torch.load(log_path + '/model_latest'+str(i)+'.t7'))
        model.eval()
        out = model(x, edges, val_index)
        pred_label = out.max(dim = 1)[1]
        val_acc_latest = acc(pred_label, y_val)
        out = model(x, edges, test_index)
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
        #    out_val = model(x, edges, val_index)
        #    out_test = model(x, edges, test_index)
        #    [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro],
        #        [test_acc_full, test_acc, test_auc, 
        #        test_f1_micro, test_f1_macro]] = link_prediction_evaluation(out_val, out_test, y_val, y_test)
        #    
        #    model.load_state_dict(torch.load(log_path + '/model_latest'+str(i)+'.t7'))
        #    model.eval()
        #    out_val = model(x, edges, val_index)
        #    out_test = model(x, edges, test_index)
        #    [[val_acc_full_latest, val_acc_latest, val_auc_latest, val_f1_micro_latest, val_f1_macro_latest],
        #                    [test_acc_full_latest, test_acc_latest, test_auc_latest, 
        #                    test_f1_micro_latest, test_f1_macro_latest]] = link_prediction_evaluation(out_val, out_test, y_val, y_test)
        #    ####################
        #    # Save testing results
        #    ####################
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
#
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
#
        #    results[i] = [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro],
        #                    [test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro],
        #                    [val_acc_full_latest, val_acc_latest, val_auc_latest, val_f1_micro_latest, val_f1_macro_latest],
        #                    [test_acc_full_latest, test_acc_latest, test_auc_latest, test_f1_micro_latest, test_f1_macro_latest]]
        #
        with open(log_path + '/log'+str(i)+'.csv', 'w') as file:
            file.write(log_str_full)
            file.write('\n')
        torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.epochs = 1

    if args.to_undirected: 
        save_name = args.method_name + 'lr' + str(int(args.lr*1000)) + 'num_filters' + 'heads' + str(int(args.heads))+  str(int(args.num_filter))+ 'task_' + args.task + '_undirected' + '_noisy' +  str(args.noisy)
    else:
        save_name = args.method_name + 'lr' + str(int(args.lr*1000)) + 'num_filters' + 'heads' + str(int(args.heads))+ str(int(args.num_filter))+ 'task_' + args.task + '_noisy' +  str(args.noisy)
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