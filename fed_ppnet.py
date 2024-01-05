import torch
from Protopnet import ProtoPNet
from utils import *
from train_or_test import *
from push_prot_chex import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Fed_PPNet():

    def __init__(self, model, clients, train_loaders, train_push_loaders, test_loaders, prot_shape, numEpoch, warmEpoch, push_start, num_round,
    model_dir, prototype_img_filename_prefix, prototype_self_act_filename_prefix, proto_bound_boxes_filename_prefix, root_dir_for_saving_prototypes, joint_optimizer_lrs, 
    joint_lr_step_size, warm_optimizer_lrs, last_layer_optimizer_lr):
        model = model.to(device) # server's model
        self.model = torch.nn.DataParallel(model)
        self.clients = clients
        self.train_loaders = train_loaders
        self.train_push_loaders = train_push_loaders
        self.test_loaders = test_loaders
        self.numEpoch = numEpoch # number of local training epochs per communication round
        self.warmEpoch = warmEpoch # number of local warm epochs
        self.push_start = push_start
        self.num_round = num_round
        self.prot_shape = prot_shape

        # initialize lists to store clients' models and optimizers
        self.name_models = list()
        self.name_joint_optim = list()
        self.name_warm_optim = list()
        self.name_last_layer_optim = list()

        # set names to save prototypes
        self.model_dir = model_dir
        self.prototype_img_filename_prefix = prototype_img_filename_prefix 
        self.prototype_self_act_filename_prefix = prototype_self_act_filename_prefix
        self.proto_bound_boxes_filename_prefix = proto_bound_boxes_filename_prefix
        self.root_dir_for_saving_prototypes = root_dir_for_saving_prototypes

        self.prototype_activation_function = 'log'
        self.joint_optimizer_lrs = joint_optimizer_lrs
        self.joint_lr_step_size = joint_lr_step_size
        self.warm_optimizer_lrs = warm_optimizer_lrs
        self.last_layer_optimizer_lr = last_layer_optimizer_lr

    # create dictionaries to keep the info about the clients' models
    def model_dict_PPNet(self, in_model_dict = None):
            if in_model_dict is None:
                model_dict = dict()
            else:
                model_dict = self.create_dict(in_model_dict)
            joint_optimizer_dict= dict()
            warm_optimizer_dict= dict()
            last_layer_optimizer_dict= dict()

            for i in range(self.clients):
                model_name ="model"+str(i)
                if in_model_dict is None:
                    model_info = ProtoPNet.construct_PPNet(base_architecture='densenet121',
                                pretrained=True,
                                img_size=224,
                                prot_shape=self.prot_shape,
                                num_classes=2,
                                prototype_activation_function='log',
                                add_on_layers_type = 'regular')
                    model_dict.update({model_name : model_info })
                else:
                    model_info = model_dict[model_name]

                joint_optimizer_name="joint_optimizer"+str(i)
                joint_optimizer_specs = \
                    [{'params': model_info.features.parameters(), 'lr': self.joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
                    {'params': model_info.add_on_layers.parameters(), 'lr': self.joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
                    {'params': model_info.prototype_vectors, 'lr': self.joint_optimizer_lrs['prototype_vectors']},
                    ]
                joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
                joint_optimizer_dict.update({joint_optimizer_name : joint_optimizer})

                warm_optimizer_name="warm_optimizer"+str(i)
                warm_optimizer_specs = \
                    [{'params': model_info.add_on_layers.parameters(), 'lr': self.warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
                    {'params': model_info.prototype_vectors, 'lr': self.warm_optimizer_lrs['prototype_vectors']},
                    ]
                warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
                warm_optimizer_dict.update({warm_optimizer_name: warm_optimizer})

                last_layer_optimizer_name="last_layer_optimizer"+str(i)
                last_layer_optimizer_specs = [{'params': model_info.last_layer.parameters(), 'lr': self.last_layer_optimizer_lr}]
                last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
                
                last_layer_optimizer_dict.update({last_layer_optimizer_name: last_layer_optimizer})

            return model_dict, joint_optimizer_dict, warm_optimizer_dict, last_layer_optimizer_dict

    '''Averaging the parameters''' #######################################################################################################
    # average prototype vectors over the clients
    def get_avg_param_prot_vectors(self, model_dict):

        prot_mean_w = torch.zeros(size = model_dict[self.name_models[0]].prototype_vectors.shape).to(device)

        with torch.no_grad():
        
            for i in range(self.clients):  
                prot_mean_w += model_dict[self.name_models[i]].prototype_vectors.data.clone().to(device)

            prot_mean_w = prot_mean_w / self.clients
            
        return prot_mean_w
    
    # average last layer weights over the clients
    def get_avg_param_last_layer(self, model_dict):

        last_mean_w = torch.zeros(size = model_dict[self.name_models[0]].last_layer.weight.shape).to(device)

        with torch.no_grad():
        
            for i in range(self.clients):  
                last_mean_w += model_dict[self.name_models[i]].last_layer.weight.data.clone().to(device)

            last_mean_w = last_mean_w / self.clients
            
        return last_mean_w 

    # average the weights and biases of added conv layers over the clients
    def get_avg_param_added_layers(self, model_dict):
        conv1_mean_w = torch.zeros(size = model_dict[self.name_models[0]].add_on_layers[0].weight.shape).to(device)
        conv1_mean_b = torch.zeros(size = model_dict[self.name_models[0]].add_on_layers[0].bias.shape).to(device)

        conv2_mean_w = torch.zeros(size = model_dict[self.name_models[0]].add_on_layers[2].weight.shape).to(device)
        conv2_mean_b = torch.zeros(size = model_dict[self.name_models[0]].add_on_layers[2].bias.shape).to(device)

        with torch.no_grad():
        
            for i in range(self.clients):
                conv1_mean_w += model_dict[self.name_models[i]].add_on_layers[0].weight.data.clone().to(device)
                conv1_mean_b += model_dict[self.name_models[i]].add_on_layers[0].bias.data.clone().to(device)

                conv2_mean_w += model_dict[self.name_models[i]].add_on_layers[2].weight.data.clone().to(device)
                conv2_mean_b += model_dict[self.name_models[i]].add_on_layers[2].bias.data.clone().to(device)

        conv1_mean_w = conv1_mean_w / self.clients
        conv1_mean_b = conv1_mean_b / self.clients

        conv2_mean_w = conv2_mean_w / self.clients
        conv2_mean_b = conv2_mean_b / self.clients

        return conv1_mean_w, conv1_mean_b, conv2_mean_w, conv2_mean_b

    # average the weights and biases of conv layers over the clients
    def get_avg_param_features(self, model_dict):
        params = [] 
        for param in model_dict[self.name_models[0]].features.features.parameters():
            size = torch.zeros(size = param.shape).to(device)
            params.append(size)
        
        with torch.no_grad():
            all_param = []
            for i in range(self.clients):
                client_param = []
                for param in model_dict[self.name_models[i]].features.features.parameters():
                    client_param.append(param)
                all_param.append(client_param)
        
            for i in all_param:
                for j in range(len(params)):
                    params[j] += i[j].data.clone().to(device)

            for j in range(len(params)):
                params[j] = params[j] / self.clients

        return params
    
    # average the running mean and var of norm layers over the clients
    def get_avg_param_norm(self, model_dict):
        layers = []
        for name, mod in model_dict[self.name_models[0]].features.features.named_modules():
            if name.split('.')[-1] == 'norm':
                layers.append(torch.zeros(size = mod.running_mean.shape).to(device))
                layers.append(torch.zeros(size = mod.running_var.shape).to(device))
            if name.split('.')[-1] == 'norm0':
                layers.append(torch.zeros(size = mod.running_mean.shape).to(device))
                layers.append(torch.zeros(size = mod.running_var.shape).to(device))
            if name.split('.')[-1] == 'norm1':
                layers.append(torch.zeros(size = mod.running_mean.shape).to(device))
                layers.append(torch.zeros(size = mod.running_var.shape).to(device))
            if name.split('.')[-1] == 'norm2':
                layers.append(torch.zeros(size = mod.running_mean.shape).to(device))
                layers.append(torch.zeros(size = mod.running_var.shape).to(device))
            if name.split('.')[-1] == 'norm5':
                layers.append(torch.zeros(size = mod.running_mean.shape).to(device))
                layers.append(torch.zeros(size = mod.running_var.shape).to(device))
        
        with torch.no_grad():
            all_param = []
            for i in range(self.clients):
                client_layers = []
                for name, mod in model_dict[self.name_models[i]].features.features.named_modules():
                    if name.split('.')[-1] == 'norm':
                        client_layers.append(mod.running_mean)
                        client_layers.append(mod.running_var)
                    if name.split('.')[-1] == 'norm0':
                        client_layers.append(mod.running_mean)
                        client_layers.append(mod.running_var)
                    if name.split('.')[-1] == 'norm1':
                        client_layers.append(mod.running_mean)
                        client_layers.append(mod.running_var)
                    if name.split('.')[-1] == 'norm2':
                        client_layers.append(mod.running_mean)
                        client_layers.append(mod.running_var)
                    if name.split('.')[-1] == 'norm5':
                        client_layers.append(mod.running_mean)
                        client_layers.append(mod.running_var)
                all_param.append(client_layers)
                
            for i in all_param:
                for j in range(len(layers)):
                    layers[j] += i[j].data.clone().to(device)

            for j in range(len(layers)):
                layers[j] = layers[j] / self.clients

        return layers
    
    '''Updating the global model's parameters''' #######################################################################################################
    # update server's prototype vectors
    def update_main_model_param_prot_vectors (self, model_dict):
        prot_mean_w = self.get_avg_param_prot_vectors(model_dict)
        with torch.no_grad():
            self.model.module.prototype_vectors.data = prot_mean_w.data.clone()  
            
        return  self.model

    # update serever's last layer weights
    def update_main_model_param_last_layer (self, model_dict):
        last_mean_w = self.get_avg_param_last_layer(model_dict)
        with torch.no_grad():
            self.model.module.last_layer.weight.data = last_mean_w.data.clone()  
            
        return  self.model
    
    # update server's weights and biases of the added conv layers
    def update_main_model_param_added_layers (self, model_dict):
        conv1_mean_w, conv1_mean_b, conv2_mean_w, conv2_mean_b = self.get_avg_param_added_layers(model_dict)
        with torch.no_grad():
            self.model.module.add_on_layers[0].weight.data = conv1_mean_w.data.clone()
            self.model.module.add_on_layers[0].bias.data = conv1_mean_b.data.clone()

            self.model.module.add_on_layers[2].weight.data = conv2_mean_w.data.clone()
            self.model.module.add_on_layers[2].bias.data = conv2_mean_b.data.clone()    
            
        return  self.model
    
    # update serever's weights and biases of the conv layers
    def update_main_model_param_features (self, model_dict):
        params = self.get_avg_param_features(model_dict)
        with torch.no_grad():
            for num, param in enumerate(self.model.module.features.features.parameters()):
                param.data = params[num].data.clone()
            
        return  self.model
    
    # update serever's running means and vars
    def update_main_model_param_norm (self, model_dict):
        layers = self.get_avg_param_norm(model_dict)
        with torch.no_grad():
            num=0
            for name, mod in self.model.module.features.features.named_modules():
                if name.split('.')[-1] == 'norm':
                    mod.running_mean.data = layers[num].data.clone()
                    num += 1
                    mod.running_var.data = layers[num].data.clone()
                    num += 1
                if name.split('.')[-1] == 'norm0':
                    mod.running_mean.data = layers[num].data.clone()
                    num += 1
                    mod.running_var.data = layers[num].data.clone()
                    num += 1
                if name.split('.')[-1] == 'norm1':
                    mod.running_mean.data = layers[num].data.clone()
                    num += 1
                    mod.running_var.data = layers[num].data.clone()
                    num += 1
                if name.split('.')[-1] == 'norm2':
                    mod.running_mean.data = layers[num].data.clone()
                    num += 1
                    mod.running_var.data = layers[num].data.clone()
                    num += 1
                if name.split('.')[-1] == 'norm5':
                    mod.running_mean.data = layers[num].data.clone()
                    num += 1
                    mod.running_var.data = layers[num].data.clone()
                    num += 1
            
        return  self.model
    

    '''Sending updated parameters to clients''' #######################################################################################################
    # send (updated) last layer parameters and prototypes to clients
    def send_main_model_to_clients(self, model_dict):
        with torch.no_grad():
            for i in range(self.clients):
                    model_dict[self.name_models[i]].last_layer.weight.data = self.model.module.last_layer.weight.data.clone()
                    model_dict[self.name_models[i]].prototype_vectors.data = self.model.module.prototype_vectors.data.clone()

        return model_dict
    
    # send (updated) model.add_on_layers to clients
    def send_main_model_added_layers_to_clients(self, model_dict):
        with torch.no_grad():
            for i in range(self.clients):
                model_dict[self.name_models[i]].add_on_layers[0].weight.data = self.model.module.add_on_layers[0].weight.data.clone()
                model_dict[self.name_models[i]].add_on_layers[0].bias.data = self.model.module.add_on_layers[0].bias.data.clone()

                model_dict[self.name_models[i]].add_on_layers[2].weight.data = self.model.module.add_on_layers[2].weight.data.clone()
                model_dict[self.name_models[i]].add_on_layers[2].bias.data = self.model.module.add_on_layers[2].bias.data.clone()

        return model_dict
    
    # send (updated) model conv layers parameters to clients
    def send_main_model_features_to_clients(self, model_dict):
        with torch.no_grad():
            for i in range(self.clients):
                for param1, param2 in zip(model_dict[self.name_models[i]].features.features.parameters(), self.model.module.features.features.parameters()):
                    param1.data = param2.data.clone()

        return model_dict
    
    # send (updated) model means and vars to clients
    def send_main_model_norm_to_clients(self, model_dict):
        with torch.no_grad():
            for i in range(self.clients):
                for (name1, mod1), (name2, mod2) in zip(model_dict[self.name_models[i]].features.features.named_modules(), self.model.module.features.features.named_modules()):
                    if name1.split('.')[-1] == 'norm':
                        mod1.running_mean.data = mod2.running_mean.data.clone()
                        mod1.running_var.data = mod2.running_var.data.clone()
                    if name1.split('.')[-1] == 'norm0':
                        mod1.running_mean.data = mod2.running_mean.data.clone()
                        mod1.running_var.data = mod2.running_var.data.clone()
                    if name1.split('.')[-1] == 'norm1':
                        mod1.running_mean.data = mod2.running_mean.data.clone()
                        mod1.running_var.data = mod2.running_var.data.clone()
                    if name1.split('.')[-1] == 'norm2':
                        mod1.running_mean.data = mod2.running_mean.data.clone()
                        mod1.running_var.data = mod2.running_var.data.clone()
                    if name1.split('.')[-1] == 'norm5':
                        mod1.running_mean.data = mod2.running_mean.data.clone()
                        mod1.running_var.data = mod2.running_var.data.clone()

        return model_dict
    

    '''Training functions''' #######################################################################################################
    def client_train_warm (self, model_dict, warm_optimizer_dict):
      for i in range(self.clients):
            train_data = self.train_loaders[i]
            test_data = self.test_loaders[i]

            ppnet = model_dict[self.name_models[i]].to(device)
            model_client = torch.nn.DataParallel(ppnet)
            warm_optimizer_client = warm_optimizer_dict[self.name_warm_optim[i]]
            print("Client", i)
            for epoch in range(self.warmEpoch):
                mode(model_client, warm=True)
                model_client.train()        
                train_accuracy, loss = train_or_test(model_client, train_data, warm_optimizer_client, class_specific=True)

                model_client.eval()
                test_accuracy, loss_te = train_or_test(model_client, test_data, class_specific=True)

                if epoch == self.warmEpoch - 1:
                    print("Epoch: {:3.0f}".format(epoch+1) + " | train accuracy: {:7.5f}".format(train_accuracy) + " | test accuracy: {:7.5f}".format(test_accuracy))  
    
    def client_train_joint (self, model_dict, joint_optimizer_dict, round):
        for i in range(self.clients):
            train_data = self.train_loaders[i]
            test_data = self.test_loaders[i]

            ppnet = model_dict[self.name_models[i]].to(device)
            model_client = torch.nn.DataParallel(ppnet)
            joint_optimizer_client = joint_optimizer_dict[self.name_joint_optim[i]]
            # joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer_client, step_size=self.joint_lr_step_size, gamma=0.1)
            print("Client", i)
            for epoch in range(self.numEpoch):
                mode(model_client, joint=True)
                model_client.train()
                # joint_lr_scheduler.step()       
                train_accuracy, loss = train_or_test(model_client, train_data, joint_optimizer_client, class_specific=True)

                model_client.eval()
                acc, loss_te = train_or_test(model_client, test_data, class_specific=True)
        
                if epoch == self.numEpoch - 1:
                    save_model_w_condition(model=model_client, model_dir=self.model_dir, model_name='client_' + str(i) + '_' + str(round) + 'nopush', acc=acc, target_acc=0.4)
                    print("Epoch: {:3.0f}".format(epoch+1) + " | train accuracy: {:7.5f}".format(train_accuracy) + " | test accuracy: {:7.5f}".format(acc))

    def clients_push_and_save(self, model_dict, round, aggregate_conv=False):
        for i in range(self.clients):
            train_push_data = self.train_push_loaders[i]

            ppnet = model_dict[self.name_models[i]].to(device)
            model_client = torch.nn.DataParallel(ppnet)
            update=True # update protoytpes by push
            if aggregate_conv:
                update=False
            push_prototypes(
                train_push_data, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=model_client, # pytorch network with prototype_vectors
                class_specific=True,
                preprocess_input_function=preprocess_input_function, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=self.root_dir_for_saving_prototypes, # if not None, prototypes will be saved here
                epoch_number=round, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix='client_' + str(i) + '_' + self.prototype_img_filename_prefix,
                prototype_self_act_filename_prefix='client_' + str(i) + '_' + self.prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix='client_' + str(i) + '_' + self.proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                update=update)

    def client_train_last (self, model_dict, last_layer_optimizer_dict, round):
        for i in range(self.clients):
            train_data = self.train_loaders[i]
            test_data = self.test_loaders[i]

            ppnet = model_dict[self.name_models[i]].to(device)
            model_client = torch.nn.DataParallel(ppnet)
            last_layer_optimizer_client = last_layer_optimizer_dict[self.name_last_layer_optim[i]]
            print("Client", i)
            if self.prototype_activation_function != 'linear':
                mode(model_client, last=True)
                for j in range(12):
                    train_acc, loss = train_or_test(model_client, train_data, last_layer_optimizer_client, class_specific=True)
                    acc, loss_te = train_or_test(model_client, test_data, class_specific=True)
                    if j == 11:
                        save_model_w_condition(model=model_client, model_dir=self.model_dir, model_name='client_' + str(i) + '_last_' + 'round_' + str(round)  + '_push', acc=acc, target_acc=0.1)
            print("train accuracy: {:7.5f}".format(train_acc) + " | test accuracy: {:7.5f}".format(acc))
    
    def save_models(self, model_dict, round):
        for i in range(self.clients):
            test_data = self.test_loaders[0]
            ppnet = model_dict[self.name_models[i]].to(device)
            model_client = torch.nn.DataParallel(ppnet)
            model_client.eval()
            acc, loss_te = train_or_test(model_client, test_data, class_specific=True)
            save_model_w_condition(model=model_client, model_dir=self.model_dir, model_name='client_' + str(i) + '_final_' + 'round_' + str(round) + '_', acc=acc, target_acc=0.1)
            print("Test accuracy after centralized update: {:7.5f}".format(acc))
        self.model.eval()
        acc, loss_te = train_or_test(self.model, test_data, class_specific=True)
        save_model_w_condition(model=self.model, model_dir=self.model_dir, model_name='server_final_round_' + str(round) + '_', acc=acc, target_acc=0.1)
        print("Test accuracy after centralized update: {:7.5f}".format(acc))


    '''Implementation''' #######################################################################################################
    def run_Fed_PPNet (self, model_dict=None, to_continue=False, aggregate_conv=False):
        model_dict, joint_optimizer_dict, warm_optimizer_dict, last_layer_optimizer_dict = self.model_dict_PPNet(in_model_dict=model_dict)
        self.name_models = list(model_dict.keys())
        self.name_joint_optim = list(joint_optimizer_dict.keys())
        self.name_warm_optim = list(warm_optimizer_dict.keys())
        self.name_last_layer_optim = list(last_layer_optimizer_dict.keys())
        for j in range(self.num_round):
            print(f'-----Round {j}-----')
            if j == 0 and to_continue is False:
                print('The model is initialized and sent to clients')
                model_dict = self.send_main_model_to_clients(model_dict)
                if aggregate_conv:
                    model_dict = self.send_main_model_features_to_clients(model_dict)
                    model_dict = self.send_main_model_added_layers_to_clients(model_dict)
                    model_dict = self.send_main_model_norm_to_clients(model_dict)
                print('----------\nClients perform training')
                print('Warm') 
                self.client_train_warm(model_dict, warm_optimizer_dict)
                print('----------\nParameters are sent to the server and aggregated')
                self.model = self.update_main_model_param_prot_vectors(model_dict)
                self.model = self.update_main_model_param_last_layer(model_dict)
                if aggregate_conv:
                    self.model = self.update_main_model_param_features(model_dict)
                    self.model = self.update_main_model_param_added_layers(model_dict)
                    self.model = self.update_main_model_param_norm(model_dict) 
                print('Updated model is sent to clients')
                model_dict = self.send_main_model_to_clients(model_dict)
                if aggregate_conv:
                    model_dict = self.send_main_model_features_to_clients(model_dict)
                    model_dict = self.send_main_model_added_layers_to_clients(model_dict)
                    model_dict = self.send_main_model_norm_to_clients(model_dict)
            else:

                # print('Server model weights of the 1st normalization layer:')
                # print(self.model.module.features.features.denseblock1.denselayer1.norm1.weight[-1])
                # print('First client model weights of the 1st normalization layer:')
                # print(model_dict['model0'].features.features.denseblock1.denselayer1.norm1.weight[-1])

                print('----------\nClients perform training')  
                self.client_train_joint(model_dict, joint_optimizer_dict, j)
                if aggregate_conv:
                    self.client_train_last(model_dict, last_layer_optimizer_dict, j)         
                
                # print('Server model weights of the 1st normalization layer:')
                # print(self.model.module.features.features.denseblock1.denselayer1.norm1.weight[-1])
                # print('First client model weights of the 1st normalization layer:')
                # print(model_dict['model0'].features.features.denseblock1.denselayer1.norm1.weight[-1])

                print('----------\nParameters are sent to the server and aggregated')
                self.model = self.update_main_model_param_prot_vectors(model_dict)
                self.model = self.update_main_model_param_last_layer(model_dict)
                if aggregate_conv:
                    self.model = self.update_main_model_param_features(model_dict)
                    self.model = self.update_main_model_param_added_layers(model_dict)
                    self.model = self.update_main_model_param_norm(model_dict)

                # print('Server model weights of the 1st normalization layer:')
                # print(self.model.module.features.features.denseblock1.denselayer1.norm1.weight[-1])
                # print('First client model weights of the 1st normalization layer:')
                # print(model_dict['model0'].features.features.denseblock1.denselayer1.norm1.weight[-1])

                print('Updated model is sent to clients')
                model_dict = self.send_main_model_to_clients(model_dict)
                if aggregate_conv:
                    model_dict = self.send_main_model_features_to_clients(model_dict)
                    model_dict = self.send_main_model_added_layers_to_clients(model_dict)
                    model_dict = self.send_main_model_norm_to_clients(model_dict)
                
                
                # print('Server model weights of the 1st normalization layer:')
                # print(self.model.module.features.features.denseblock1.denselayer1.norm1.weight[-1])
                # print('First client model weights of the 1st normalization layer:')
                # print(model_dict['model0'].features.features.denseblock1.denselayer1.norm1.weight[-1])
                # print('Second client model weights of the 1st normalization layer:')
                # print(model_dict['model1'].features.features.denseblock1.denselayer1.norm1.weight[-1])

                self.clients_push_and_save(model_dict, j, aggregate_conv=aggregate_conv)
                if aggregate_conv:
                    self.save_models(model_dict, j)
                else:
                    self.client_train_last(model_dict, last_layer_optimizer_dict, j)

        return model_dict