import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from receptive_field import compute_proto_layer_rf_info_v2
from vgg_features import vgg19_features
from resnet_features import resnet34_features
from densenet_features import densenet121_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_architecture_to_features = {'vgg19': vgg19_features, 'resnet34': resnet34_features, 'densenet121': densenet121_features}

class ProtoPNet(nn.Module):

    def __init__(self, features, img_size, prot_shape, proto_layer_rf_info,
                 num_classes, fed, init_weights=True,
                 prototype_activation_function='log', add_on_layers_type='bottleneck') -> None:
        super(ProtoPNet, self).__init__()
        self.features = features
        self.img_size = img_size 
        self.prot_shape = prot_shape
        self.num_classes = num_classes
        self.num_prot = prot_shape[0]
        self.prototype_activation_function = prototype_activation_function
        self.epsilon = 1e-4
        self.fed = fed

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prot % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prot_class_id = torch.zeros(self.num_prot, self.num_classes).to(device)

        num_prot_per_class = self.num_prot // self.num_classes
        for j in range(self.num_prot):
            self.prot_class_id[j, j // num_prot_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info #what is this?

        in_channels = [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels

        # self.add_on_layers = nn.Sequential(
        #         nn.Conv2d(in_channels=in_channels, out_channels=self.prot_shape[1], kernel_size=1),
        #         nn.ReLU(),
        #         nn.Conv2d(in_channels=self.prot_shape[1], out_channels=self.prot_shape[1], kernel_size=1),
        #         nn.Sigmoid())

        # this has to be named features to allow the precise loading

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prot_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prot_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prot_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prot_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prot_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prot_shape[1], out_channels=self.prot_shape[1], kernel_size=1),
                nn.Sigmoid()
                )

        self.prototype_vectors = nn.Parameter(torch.rand(self.prot_shape), requires_grad=True)

        # self.in_feature_w = nn.Parameter(torch.ones(self.prot_shape), requires_grad=True)
        # self.in_feature_b = nn.Parameter(torch.zeros(self.prot_shape), requires_grad=True)

        # self.lin_feature_w = nn.Parameter(torch.ones(self.prot_shape), requires_grad=True)
        # self.lin_feature_b = nn.Parameter(torch.zeros(self.prot_shape), requires_grad=True)
        
        # self.out_feature_w = nn.Parameter(torch.ones(1, self.num_classes), requires_grad=True)
        # self.out_feature_b = nn.Parameter(torch.zeros(1, self.num_classes), requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prot_shape), requires_grad=False)

        self.last_layer = nn.Linear(self.num_prot, self.num_classes, bias=False)

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        ''' the feature input to prototype layer'''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x
    
    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)
    
    def apply_in_and_lin(self):
        self.prototype_vectors = (self.in_feature_b + self.prototype_vectors) * self.in_feature_w
        self.prototype_vectors = (self.lin_feature_b + self.prototype_vectors) * self.lin_feature_w
        return self.prototype_vectors

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)
        # if self.fed:
        #     self.prototype_vectors = self.apply_in_and_lin()
        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors) #shape(N*P*7*7)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)
        return distances

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        distances = self.prototype_distances(x)
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling
        min_distances = -F.max_pool2d(-distances, # why?? okay
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3])) ##shape(N*P*1*1)
        min_distances = min_distances.view(-1, self.num_prot) ##(N*P)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)
        # if self.fed:
        #     logits = (self.out_feature_b + logits) * self.out_feature_w
        return logits, min_distances

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prot_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prot_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prot_class_id)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def construct_PPNet(base_architecture, pretrained=True, img_size=224,
                    prot_shape=(200, 128, 1, 1), num_classes=10,
                    prototype_activation_function='log',
                    add_on_layers_type='regular', fed=False):
        # features = base_architecture
        features = base_architecture_to_features[base_architecture](pretrained=pretrained)
        layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
        # layer_filter_sizes, layer_strides, layer_paddings = 5, 1, 2
        proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                            layer_filter_sizes=layer_filter_sizes,
                                                            layer_strides=layer_strides,
                                                            layer_paddings=layer_paddings,
                                                            prototype_kernel_size=prot_shape[2])
        return ProtoPNet(features=features,
                    img_size=img_size,
                    prot_shape=prot_shape,
                    proto_layer_rf_info=proto_layer_rf_info,
                    num_classes=num_classes,
                    fed = fed,
                    init_weights=True,
                    prototype_activation_function=prototype_activation_function,
                    add_on_layers_type=add_on_layers_type)