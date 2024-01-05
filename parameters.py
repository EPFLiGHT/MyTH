img_size = 224
prototype_activation_function = 'log'

train_batch_size = 80
test_batch_size = 127
train_push_batch_size = 75

mean = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
std = [0.229, 0.224, 0.225]

prototype_img_filename_prefix = 'prot_img'
prototype_self_act_filename_prefix = 'prot_self_act'
proto_bound_boxes_filename_prefix = 'prot_bb'
model_dir = 'ppnet_chest/'
root_dir_for_saving_prototypes = 'prot_chest/'

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4