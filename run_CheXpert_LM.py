import os
import argparse

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import *
from Protopnet import ProtoPNet
from train_or_test import *
from push_prot_chex import *
from parameters import *

seed = 12
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser() 
parser.add_argument("--num_classes", '-nc', default=2, required=False, help="Number of classes")
parser.add_argument("--epochs", '-e', type=str, required=True, help="Number of training epochs")
parser.add_argument("--class_name", '-c', type=str, required=True, help="Name of a class") # cardiomegaly or effusion
parser.add_argument("--distribute", '-d', action='store_true', help="Set for distributing the data among the clients")
parser.add_argument("--num_clients", '-ncl', default=4, required=False, help="Number of clients")
parser.add_argument("--client_to_train", '-t', type=str, required=True, help="Number of the client to train") # please start counting from 0
parser.add_argument("--biased", '-b', action='store_true', help="Set for adding bias")
args = parser.parse_args()

num_classes = int(args.num_classes)
shape0 = 10*num_classes
prototype_shape = (shape0, 128, 1, 1)
num_clients = int(args.num_clients)

normalize = transforms.Normalize(mean=mean,
                                 std=std)

num_train_epoch = int(args.epochs)
num_warm_epoch = 5
push_epochs = np.linspace(10, 100, 10)[:-1]
push_start = 10

if not os.path.exists('pretrained_models'):
    os.mkdir('pretrained_models')
if not os.path.exists('ppnet_chest'):
    os.mkdir('ppnet_chest')
if not os.path.exists('prot_chest'):
    os.mkdir('prot_chest')

if __name__ == '__main__':
    if args.distribute:
        data_path = args.class_name + '/'
        train_dir = data_path + 'train/'
        test_dir = data_path + 'test/'
        train_push_dir = data_path + 'push/'
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

    data_path = 'client_' + args.client_to_train + '/'
    
    # Add a synthetic bias to one client's dataset
    if args.biased and args.class_name == 'cardiomegaly':
        num_client = 3 # we always add bias to the 4th client
        unicode = '\U0001F42D'
        bias_folder = 'positive'
        size = 35
        percent = 100
        adding_emoji(num_client, unicode, bias_folder, size, percent)
    elif args.biased and args.class_name == 'effusion':
        data_path = 'drains/'

    train_dir = data_path + 'train/'
    test_dir = data_path + 'test/'
    train_push_dir = data_path + 'push/'

    # train set
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=2, pin_memory=False)
    # push set
    train_push_dataset = datasets.ImageFolder(
        train_push_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=2, pin_memory=False)
    # test set
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=2, pin_memory=False)
    
    # Centralized training on a local set

    # define a model
    ppnet = ProtoPNet.construct_PPNet(base_architecture='densenet121',
                                pretrained=True,
                                img_size=img_size,
                                prot_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type = 'regular')

    ppnet = ppnet.to(device)
    model = torch.nn.DataParallel(ppnet)

    joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    # joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

    warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # Training
    for epoch in range(num_train_epoch):
        print('epoch', epoch)

        if epoch < num_warm_epoch:
            mode(model, warm=True)
            model.train()
            correct_ratio, loss = train_or_test(model, train_loader, warm_optimizer, class_specific=True)

        else:
            mode(model, joint=True)
            model.train()
            # joint_lr_scheduler.step()
            _, loss = train_or_test(model, train_loader, joint_optimizer, class_specific=True)

        model.eval()
        print('---test---')
        acc, loss_test = train_or_test(model, test_loader, class_specific=True)
        save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', acc=acc, target_acc=0.60)

        if epoch >= push_start and epoch in push_epochs:
            push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=model, # pytorch network with prototype_vectors
                class_specific=True,
                preprocess_input_function=preprocess_input_function, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=root_dir_for_saving_prototypes, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True)

            model.eval()
            acc, loss_test = train_or_test(model, test_loader, class_specific=True)

            if prototype_activation_function != 'linear':
                mode(model, last=True)
                for i in range(12):
                    print('iteration: \t{0}'.format(i))
                    _, loss = train_or_test(model, train_loader, last_layer_optimizer, class_specific=True)
                    acc, loss_test = train_or_test(model, test_loader, class_specific=True)
                    save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', acc=acc, target_acc=0.60)




