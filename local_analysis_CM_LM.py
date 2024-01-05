import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from utils import *

import os
from utils import *
from train_or_test import *
from parameters import *
import cv2
import copy
import re

seed = 12
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# specify the test image to be analyzed
test_image_dir = ''
test_image_name = '10.jpg'
test_image_label = 1
save_analysis_path = test_image_dir
test_image_path = os.path.join(test_image_dir, test_image_name)

# load the model
check_test_accu = False
load_model_dir = 'Pleural_effusion/Local_drain/ppnet_chest/'
load_model_name = '20_11push0.8901.pth'
load_model_path = os.path.join(load_model_dir, load_model_name)

epoch_number_str = re.search(r'\d+', load_model_name).group(0) #for CM and LM
start_epoch_number = int(epoch_number_str)

print('load model from ' + load_model_path)

ppnet = torch.load(load_model_path, map_location=torch.device('cpu'))
ppnet = ppnet.to(device)

img_size = ppnet.img_size
prototype_shape = ppnet.prot_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

normalize = transforms.Normalize(mean=mean,
                                 std=std)

##### SANITY CHECK
# confirm prototype class identity
load_img_dir = 'Pleural_effusion/Local_drain/prot_chest/' # directory with saved prototypes

prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'prot_bb'+epoch_number_str+'.npy'))
prototype_img_identity = prototype_info[:, -1]

print('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
print('Their class identities are: ' + str(prototype_img_identity))

# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()
if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prot:
    print('All prototypes connect most strongly to their respective classes.')
else:
    print('WARNING: Not all prototypes connect most strongly to their respective classes.')
print(np.sum(prototype_max_connection == prototype_img_identity))
print(ppnet.num_prot)
print(prototype_img_identity)

##### HELPER FUNCTIONS FOR PLOTTING
def preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def preprocess_input_function(x):
    '''
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    '''
    return preprocess(x, mean=mean, std=std)

def undo_preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y

def undo_preprocess_input_function(x):
    '''
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    '''
    return undo_preprocess(x, mean=mean, std=std)

def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    
    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img

def save_prototype(fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prot_img'+str(index)+'.png'))
    #plt.axis('off')
    plt.imsave(fname, p_img)


def save_prototype_self_activation(fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                    'prot_img-original_with_self_act'+str(index)+'.png'))
    #plt.axis('off')
    plt.imsave(fname, p_img)

def save_prototype_original_img_with_bbox(fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prot_img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=1)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(fname, p_img_rgb)

def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=1)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imsave(fname, img_rgb_float)

# load the test image and forward it through the network
preprocess = transforms.Compose([
   transforms.Resize((img_size,img_size)),
   transforms.ToTensor(),
   normalize
])

img_pil = Image.open(test_image_path).convert("RGB")
img_tensor = preprocess(img_pil)

img_variable = Variable(img_tensor.unsqueeze(0))

# images_test = img_variable
labels_test = torch.tensor([test_image_label])

logits, min_distances = ppnet(img_variable)
conv_output, distances = ppnet.push_forward(img_variable)
prototype_activations = ppnet.distance_2_similarity(min_distances)
prototype_activation_patterns = ppnet.distance_2_similarity(distances)
if ppnet.prototype_activation_function == 'linear':
    prototype_activations = prototype_activations + max_dist
    prototype_activation_patterns = prototype_activation_patterns + max_dist

tables = []
for i in range(logits.size(0)):
    tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
    print(str(i) + ' ' + str(tables[-1]))

idx = 0
predicted_cls = tables[idx][0]
correct_cls = tables[idx][1]
print('Predicted: ' + str(predicted_cls))
print('Actual: ' + str(correct_cls))
original_img = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'),
                                     img_variable, idx)

##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))

print('Most activated 10 prototypes of this image:')
array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
for i in range(1,11):
    print('top {0} activated prototype for this image:'.format(i))
    save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'top-%d_activated_prototype.png' % i),
                   start_epoch_number, sorted_indices_act[-i].item())
    save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                             'top-%d_activated_prototype_in_original_pimg.png' % i),
                                          epoch=start_epoch_number,
                                          index=sorted_indices_act[-i].item(),
                                          bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                                          bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                                          bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                                          bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                                          color=(0, 255, 255))
    save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                'top-%d_activated_prototype_self_act.png' % i),
                                   start_epoch_number, sorted_indices_act[-i].item())
    print('prototype index: {0}'.format(sorted_indices_act[-i].item()))
    print('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
    if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
        print('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
    print('activation value (similarity score): {0}'.format(array_act[-i]))
    print('last layer connection with predicted class: {0}'.format((torch.max(ppnet.last_layer.weight, dim=0)[0])[sorted_indices_act[-i].item()]))
    
    activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
    upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                              interpolation=cv2.INTER_CUBIC)
    
    # show the most highly activated patch of the image by this prototype
    high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
    high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                  high_act_patch_indices[2]:high_act_patch_indices[3], :]
    
    print('most highly activated patch of the chosen image by this prototype:')
    #plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'most_highly_activated_patch_by_top-%d_prototype.png' % i),
               high_act_patch)
    print('most highly activated patch by this prototype shown in the original image:')
    imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                     img_rgb=original_img,
                     bbox_height_start=high_act_patch_indices[0],
                     bbox_height_end=high_act_patch_indices[1],
                     bbox_width_start=high_act_patch_indices[2],
                     bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
    
    # show the image overlayed with prototype activation map
    rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
    rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[...,::-1]
    overlayed_img = 0.5 * original_img + 0.3 * heatmap
    print('prototype activation map of the chosen image:')
    #plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'prototype_activation_map_by_top-%d_prototype.png' % i),
               overlayed_img)
    print('--------------------------------------------------------------')


if predicted_cls == correct_cls:
    print('Prediction is correct.')
else:
    print('Prediction is wrong.')