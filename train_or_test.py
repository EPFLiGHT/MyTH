import torch
import torch.nn.functional as F
from Protopnet import ProtoPNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_or_test(model, dataloader, optimizer=None, coefs=None, class_specific=False, use_l1_mask=False):
    is_train = optimizer is not None
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    # total_loss = 0

    for i, (image, label) in enumerate(dataloader):
        input = image.to(device)
        target = label.to(device)

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, min_distances = model(input) #how do we return the distances?
            cross_entropy = F.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prot_shape[1] #module
                            * model.module.prot_shape[2]
                            * model.module.prot_shape[3]) #what is this??

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes (N*P)
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prot_class_id[:,label]).to(device) #module
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
                    l1_mask = 1 - torch.t(model.module.prot_class_id).to(device) #module
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1) 

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1) #module

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

            # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            # total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # print('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    # print('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    # if class_specific:
    #     print('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    #     print('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    print('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    # print('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    # p = model.prototype_vectors.view(model.module.num_prot, -1).cpu()
    # with torch.no_grad():
    #     p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    # print('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))
    
    return n_correct / n_examples, total_cross_entropy / n_batches

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def mode(model, warm=False, joint=False, last=False, fed=False):
    for p in model.module.last_layer.parameters(): #module
            p.requires_grad = True

    # if not fed:
    #     model.module.in_feature_w.requires_grad = False
    #     model.module.in_feature_b.requires_grad = False
    #     model.module.out_feature_w.requires_grad = False
    #     model.module.out_feature_b.requires_grad = False
    #     model.module.lin_feature_w.requires_grad = False
    #     model.module.lin_feature_b.requires_grad = False


    if warm:
        for p in model.module.features.parameters():
            p.requires_grad = False
        for p in model.module.add_on_layers.parameters():
            p.requires_grad = True
        model.module.prototype_vectors.requires_grad = True
        print('\twarm')
    elif joint:
        for p in model.module.features.parameters():
            p.requires_grad = True
        for p in model.module.add_on_layers.parameters():
            p.requires_grad = True
        model.module.prototype_vectors.requires_grad = True
        print('\tjoint')
    elif last:
        for p in model.module.features.parameters():
            p.requires_grad = False
        for p in model.module.add_on_layers.parameters():
            p.requires_grad = False
        model.module.prototype_vectors.requires_grad = False
        print('\tlast layer')
            
    # elif fed:
    #     for p in model.module.features.parameters():
    #         p.requires_grad = False
    #     for p in model.module.add_on_layers.parameters():
    #         p.requires_grad = False
    #     model.module.prototype_vectors.requires_grad = False