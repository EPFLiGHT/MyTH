import gzip
import numpy as np
import torch
import os
import shutil
import math
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_torch (images, labels):
    features = torch.from_numpy(images).to(device)
    labels = torch.from_numpy(labels).to(device)
    return features, labels

def standardize_data (features, mean=None, std=None):
  if mean is None and std is None:
    mean, std = features.float().mean(), features.float().std()

  features_std = features.float().sub_(mean).div_(std)
  features_std = features_std.reshape(-1, 1, 28, 28)

  return features_std, mean, std

def accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def save_model_w_condition(model, model_dir, model_name, acc, target_acc):
    '''
    model: this is not the multigpu model
    '''
    if acc > target_acc:
        # print('\tabove {0:.2f}%'.format(target_acc * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(acc)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(acc)))

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

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

def distribute_data(src_dir, seed, num_clients):
  classes = os.listdir(src_dir)
  save_path = src_dir.split('/')[-2]
  for class_name in classes:
    amount = len(os.listdir(src_dir + class_name))
    data_per_client = math.floor(amount/num_clients)
    np.random.seed(seed)
    label_inx = np.random.permutation(amount)
    for client in range(num_clients):
      client_samples = [os.listdir(src_dir + class_name)[x] for x in label_inx[client*data_per_client : (client+1)*data_per_client]]
      for j in range(data_per_client):
        shutil.copy(src_dir + class_name + '/' + client_samples[j], f'client_{client}/' + save_path + '/' + class_name)

def adding_emoji(client, unicode, bias_folder, size, percent):
  dir_train = f'client_{client}/train/{bias_folder}/'
  dir_push = f'client_{client}/push/{bias_folder}/'
  dir_test = f'client_{client}/test/{bias_folder}/'
  for dir in (dir_train, dir_push, dir_test):
    num_imgs = len(os.listdir(dir))
    inds = np.random.permutation(num_imgs)
    ind_max = int(np.ceil(num_imgs * percent / 100))
    print(ind_max)
    inds_to_bias = inds[:ind_max]
    for num, img_name in enumerate(os.listdir(dir)):
        if num in inds_to_bias:
          image = cv.imread(dir + img_name, cv.IMREAD_COLOR)
          image = Image.fromarray(np.uint8(image))
          draw = ImageDraw.Draw(image)
          path = 'NotoEmoji-VariableFont_wght.ttf' # define path to the emoji font

          font = ImageFont.truetype(path, size)
          pos = size
          draw.text((pos, 0), unicode, font=font, fill=(0,0,255))
          image = np.array(image)
          cv.imwrite(dir + img_name, image)

def evaluate(model, dataloader, coefs=None, class_specific=False, use_l1_mask=False):
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    f1 = 0
    acc_multi = 0
    # total_loss = 0
    targets = []
    outputs = []
    CM = 0

    for i, (image, label) in enumerate(dataloader):
        input = image.to(device)
        target = label.to(device)
        targets.append(target.cpu())

        with torch.no_grad():
            model.eval()
            output, min_distances = model(input) #how do we return the distances?
            cross_entropy = F.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.prot_shape[1] #module
                            * model.prot_shape[2]
                            * model.prot_shape[3]) #what is this??

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes (N*P)
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.prot_class_id[:,label]).to(device) #module
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1) # (1*N)
                # distances, _ = torch.min((min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)
                # cluster_cost_1 = torch.mean(distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes) # maybe change

                # calculate avg separation cost
                avg_separation_cost = torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)

                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.prot_class_id).to(device) #module
                    l1 = (model.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.last_layer.weight.norm(p=1) 

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.last_layer.weight.norm(p=1) #module

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            outputs.append(predicted.cpu())
            CM+=confusion_matrix(target.cpu(), predicted.cpu(),labels=[0,1])

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()
        
    targets = torch.cat(targets)
    outputs = torch.cat(outputs)
    f1 += multiclass_f1_score(outputs, targets, num_classes=2, average=None)
    acc_multi += multiclass_accuracy(outputs, targets, num_classes=2, average=None)
    score = roc_auc_score(targets, outputs, average='weighted')
    tn=CM[0][0]
    tp=CM[1][1]
    fp=CM[0][1]
    fn=CM[1][0]
    sensitivity=tp/(tp+fn)
    specificity=tn/(tn+fp)
    score = (sensitivity + specificity)/2
    
    return n_correct / n_examples, f1, acc_multi, sensitivity, specificity, score