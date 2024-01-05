import os
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torchvision.datasets as datasets

import torchvision.transforms as transforms

from utils import *
from Protopnet import ProtoPNet
from train_or_test import *
from push_prot_chex import *
from fed_ppnet import Fed_PPNet
from parameters import *

seed = 12
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser() 
parser.add_argument("--num_classes", '-nc', default=2, required=False, help="Number of classes")
parser.add_argument("--class_name", '-c', type=str, required=True, help="Name of a class") # cardiomegaly or effusion
parser.add_argument("--distribute", '-d', action='store_true', help="Set for distributing the data among the clients")
parser.add_argument("--num_clients", '-ncl', default=4, required=False, help="Number of clients")
parser.add_argument("--num_rounds", '-nr', type=int, required=True, help="Number of communication rounds")
parser.add_argument("--biased", '-b', action='store_true', help="Set for adding emoji")
parser.add_argument("--aggconv", '-agc', action='store_true', help="Set for aggregating convolutional parameters")
parser.set_defaults(biased=False)
parser.set_defaults(aggconv=False)
args = parser.parse_args()

num_classes = int(args.num_classes)
shape0 = 10*num_classes
prototype_shape = (shape0, 128, 1, 1)
num_clients = int(args.num_clients)
num_rounds = args.num_rounds

normalize = transforms.Normalize(mean=mean,
                                 std=std)

data_path = args.class_name + '/'
train_dir = data_path + 'train/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'push/'

if not os.path.exists('pretrained_models'):
    os.mkdir('pretrained_models')
if not os.path.exists('ppnet_chest'):
    os.mkdir('ppnet_chest')
if not os.path.exists('prot_chest'):
    os.mkdir('prot_chest')

if __name__ =='__main__':
    b=0 # needed for setting drains bias below
    if args.distribute:
        dir_names = os.listdir(train_dir)
        for client in range(num_clients):
            os.mkdir(f'client_{client}')
            os.mkdir(f'client_{client}/' + 'train/')
            os.mkdir(f'client_{client}/' + 'push/')
            os.mkdir(f'client_{client}/' + 'test/')
            for class_name in dir_names:
                os.mkdir(f'client_{client}/'+ 'train/' + class_name)
                os.mkdir(f'client_{client}/'+ 'push/' + class_name)
                os.mkdir(f'client_{client}/'+ 'test/' + class_name)
        
        print('Train data is being distributed')
        distribute_data(train_dir, seed, num_clients)
        print('Train push data is being distributed')
        distribute_data(train_push_dir, seed, num_clients)
        print('Test data is being distributed')
        distribute_data(test_dir, seed, num_clients)

    # Add a synthetic bias to one client's dataset
    if args.biased and args.class_name == 'cardiomegaly':
        num_client = 3 # we always add bias to the 4th client
        unicode = '\U0001F42D'
        bias_folder = 'positive'
        size = 35
        percent = 100
        adding_emoji(num_client, unicode, bias_folder, size, percent)
    elif args.biased and args.class_name == 'effusion':
        b=1

    train_datasets, train_loaders = [],[]
    train_push_datasets, train_push_loaders = [],[]
    test_datasets, test_loaders = [],[]

    for client in range(num_clients-b):
        # train set
        train_dir = f'client_{client}/' + 'train/'
        train_push_dir = f'client_{client}/' + 'push/'
        test_dir = f'client_{client}/' + 'test/'

        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
        train_datasets.append(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True,
            num_workers=2, pin_memory=False)
        train_loaders.append(train_loader)

        # push set
        train_push_dataset = datasets.ImageFolder(
            train_push_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
            ]))
        train_push_datasets.append(train_push_dataset)

        train_push_loader = torch.utils.data.DataLoader(
            train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
            num_workers=2, pin_memory=False)
        train_push_loaders.append(train_push_loader)

        # test set
        test_dataset = datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
        test_datasets.append(test_dataset)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=2, pin_memory=False)
        test_loaders.append(test_loader)
    
    if len(train_datasets) != num_clients: # we need to add a biased client
        # train set
        train_dir = f'drains/' + 'train/'
        train_push_dir = f'drains/' + 'push/'
        test_dir = f'drains/' + 'test/'

        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
        train_datasets.append(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True,
            num_workers=2, pin_memory=False)
        train_loaders.append(train_loader)

        # push set
        train_push_dataset = datasets.ImageFolder(
            train_push_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
            ]))
        train_push_datasets.append(train_push_dataset)

        train_push_loader = torch.utils.data.DataLoader(
            train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
            num_workers=2, pin_memory=False)
        train_push_loaders.append(train_push_loader)

        # test set
        test_dataset = datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
        test_datasets.append(test_dataset)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=2, pin_memory=False)
        test_loaders.append(test_loader)


    print('Data are ready!')
    # train
    model = ProtoPNet.construct_PPNet(base_architecture='densenet121',
                              pretrained=True,
                              img_size=img_size,
                              prot_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type = 'regular')
    # we set these parameters by default
    numEpoch = 10
    warmEpoch = 5
    push_start = 9
    batch_train = 80
    batch_test = 100

    # set names for directories to save models and prototypes
    model_dir = 'ppnet_chest/'
    prototype_img_filename_prefix = 'prot_img'
    prototype_self_act_filename_prefix = 'prot_self_act'
    proto_bound_boxes_filename_prefix = 'prot_bb'
    root_dir_for_saving_prototypes = 'prot_chest/'

    fed_ppnet = Fed_PPNet(model, num_clients, train_loaders, train_push_loaders, test_loaders, prototype_shape, numEpoch, warmEpoch, push_start, num_rounds,
                        model_dir, prototype_img_filename_prefix, prototype_self_act_filename_prefix,
                         proto_bound_boxes_filename_prefix, root_dir_for_saving_prototypes, joint_optimizer_lrs, joint_lr_step_size, warm_optimizer_lrs, last_layer_optimizer_lr)
    
    model_dict = fed_ppnet.run_Fed_PPNet(aggregate_conv=args.aggconv)