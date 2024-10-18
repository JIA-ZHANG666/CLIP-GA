import cv2
import os
import torch
import os.path as osp
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib
from imutils import visual_debug
from clip_utils import clip_forward
from clip_loss import SimMaxLoss, SimMinLoss, BackgroundSuppressionLoss
import voc12.dataloader
from misc import pyutils, torchutils
import os, math

from graph.graph_layer import *
from graph import *
from graph.voc_data import *
import net.resnet50_clims
from net.resnet50_clims import *
import graph.graph_layer
import time
import tracemalloc
#from torch_geometric.utils import dense_to_sparse


with open("./graph/CM_kg_57_info.json","rb") as f:
        info = json.load(f)
        KF_All_VOC_info = info['KG_VOC_info']
        
        graph_adj_mat = np.asarray(KF_All_VOC_info['S'])
graph_adj_mat = torch.FloatTensor(graph_adj_mat).cuda()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def regularization_loss(features, reg_lambda=0.1):
    # L2 regularization loss
    loss = reg_lambda * torch.norm(features, p=2)
    return loss

new_class_names = ['aeroplane', 'bicycle', 'bird avian', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair seat', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person with clothes,people,human',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor screen',
                   ]

#new_class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
            #'dog',
            #'horse', 'motorbike', 'player', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss')))

    return


# GLOBAL_SEED = 2
# import numpy as np
# import random
# def set_seed(seed):
#     print('11')
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
# GLOBAL_WORKER_ID = None
# def worker_init_fn(worker_id):
#     global GLOBAL_WORKER_ID
#     GLOBAL_WORKER_ID = worker_id
#     set_seed(GLOBAL_SEED + worker_id)
start_time = 0
end_time = 0
train_time = 0
def run(args):
    start_time = time.time()
    print("#########start_time:",start_time)
    model = getattr(importlib.import_module(args.clims_network), 'CLIMS')(n_classes=20)

    # initialize backbone network with baseline CAM
    model.load_state_dict(torch.load('cam-baseline-voc12/res50_cam.pth'), strict=True)

    ################################################
    voc_data = get_voc_data()
    graph_learner = graph.graph_layer.GraphLearner(**voc_data)
    graph_learner = torch.nn.DataParallel(graph_learner).cuda()
    graph_learner.train()

    SemanticLoss = net.resnet50_clims.SemanticContextLoss(20, 256)
    SemanticLoss = torch.nn.DataParallel(SemanticLoss).cuda()
    SemanticLoss.train()

    recam_predictor = net.resnet50_clims.Class_Predictor(20, 256)
    recam_predictor = torch.nn.DataParallel(recam_predictor).cuda()
    recam_predictor.train()

    graph_fusion = graph.graph_layer.GraphFusion(256, 20)
    graph_fusion = torch.nn.DataParallel(graph_fusion).cuda()
    graph_fusion.train()
    ##########################################

    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.clims_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.clims_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.clims_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': graph_learner.parameters(), 'lr': 0.0001, 'weight_decay': args.cam_weight_decay},#0.0001-0.5933
        {'params': recam_predictor.parameters(), 'lr': 0.1, 'weight_decay': args.cam_weight_decay},
        {'params': SemanticLoss.parameters(), 'lr': 0.0001, 'weight_decay': args.cam_weight_decay},#0.4
        {'params': graph_fusion.parameters(), 'lr': 0.0001, 'weight_decay': args.cam_weight_decay},
    ], lr=args.clims_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    # Loss
    hyper = [float(h) for h in args.hyper.split(',')]
    OTMLoss = SimMaxLoss()
    BTMLoss = SimMinLoss()
    CBSLoss = BackgroundSuppressionLoss(dname='voc')
    print(hyper)

    # CLIP
    import clip
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(args.clip, device=device)
    # for p in clip_model.parameters():
    #     p.requires_grad = False
    #for name, param in clip_model.named_parameters():
        #if "graph_learner" not in name:
            #param.requires_grad_(False)

    #for param in graph_learner.parameters():
        #param.requires_grad_(True)

    clip_model.eval()

    if args.clip == 'RN50x4':
        clip_input_size = 288
    else:
        clip_input_size = 224

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

######################################
    def zeroshot_classifier(classnames, templates, model):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [template.format(classname) for template in templates] #format with class
                texts = clip.tokenize(texts).to(device) #tokenize
                class_embeddings = model.encode_text(texts)#.detach() #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        return zeroshot_weights.t()

    fg_text_features1 = zeroshot_classifier(new_class_names, ['a photo of a {}.'], clip_model)#fg_text_features: torch.Size([20, 512])
    #fg_text_features1 = zeroshot_classifier(new_class_names, ['a clean origami {}.'], clip_model)
    fg_text_features = graph_learner(fg_text_features1, graph_adj_mat)#.detach()

###############################################

    # transform multi-hot label to class index label
    def preprocess(labels):
        new_labels = []
        for n in range(labels.size(0)):
            for idx in range(0, labels.size(1)):
                temp = torch.zeros(1, labels.size(1)).long()
                if labels[n, idx] == 1:
                    temp[0, idx] = 1
                new_labels.append(temp)
        return torch.cat(new_labels, dim=0).cuda()

    hyper = [float(h) for h in args.hyper.split(',')]
    for ep in range(args.clims_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.clims_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']#torch.Size([16, 3, 512, 512])
            img = img.cuda()
            label = pack['label'].cuda(non_blocking=True)#([16, 20])
            
            fg_label = preprocess(label.cpu())#([320, 20])
            

            x,logist = model(img)
            N, _, _, _ = x.size()
            optimizer.zero_grad()

            # foreground indices
            ########label.reshape(-1):([320])
            fg_indices = torch.nonzero(label.reshape(-1) == 1, as_tuple=False).squeeze()#([26])
          
            g_cam_224 = F.interpolate(x, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)#g_cam_224: torch.Size([16, 20, 224, 224])
           
            cam_224 = F.interpolate(x, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True).reshape(N * 20, 1, clip_input_size,
                                                                                                clip_input_size)#torch.Size([320, 1, 224, 224])
            img_224 = F.interpolate(img, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)#torch.Size([16, 3, 224, 224])
          

            fg_224_eval = []
            bg_224_eval = []
            gc_fg_224_eval = []
            gc_bg_224_eval = []
            fg_img_224 = []
            total_loss = 0
            bg_total_loss = 0
            gc_eval = []
            fg_text_eval = []
            
            temp_idx = torch.nonzero(label == 1, as_tuple=False)#([26, 2])
            for j in range(temp_idx.shape[0]):#26
                fg_224_eval.append(cam_224[fg_indices[j]] * img_224[temp_idx[j, 0]])#fg_indices[j]: torch.Size([]) temp_idx[j, 0]: torch.Size([])
                bg_224_eval.append((1 - cam_224[fg_indices[j]]) * img_224[temp_idx[j, 0]])


###############################################################################
            for class_idx in range(20):
                class_indices = torch.nonzero(label[:, class_idx] == 1).squeeze()
        
                if class_indices.numel() == 0:
                    # 如果没有图像属于该类，跳过该类
                    gc_fg_224_eval.append(torch.zeros(3, 224, 224).to(device))
                    gc_bg_224_eval.append(torch.zeros(3, 224, 224).to(device))
        
                else:
                    selected_idx = class_indices.item() if class_indices.dim() == 0 else class_indices[0]
        
                    gc_fg_224_eval.append((cam_224[selected_idx * 20 + class_idx] * img_224[selected_idx]).squeeze(0))
                    gc_bg_224_eval.append(((1 - cam_224[selected_idx * 20 + class_idx]) * img_224[selected_idx]).squeeze(0))
    
            gc_fg_224_eval = torch.stack(gc_fg_224_eval, dim=0)
            gc_bg_224_eval = torch.stack(gc_bg_224_eval, dim=0)
            clip_features_fg1 = clip_model.encode_image(gc_fg_224_eval)#.detach()
            #print("########clip_features_fg:",clip_features_fg.shape)
            clip_features_bg1 = clip_model.encode_image(gc_bg_224_eval)#.detach()
            
            fg_features, bg_features = generate_clip_input(img_224, g_cam_224, label, clip_model, clip_input_size, device)
            for i in range(img.shape[0]):
                
                edges = cosine_similarity(fg_features[i])#0.4*loss_fn-0.5946
                edges = torch.tensor(edges, dtype=torch.float).cuda()
               
                image_features = graph_learner(fg_features[i], edges)
                fg_text_features2, image_features = graph_fusion(fg_text_features, image_features)
                
                gc_eval.append(image_features)
                fg_text_eval.append(fg_text_features2)
 ###############################################################################               

            fg_224_eval = torch.stack(fg_224_eval, dim=0)#torch.Size([26?, 3, 224, 224])
            bg_224_eval = torch.stack(bg_224_eval, dim=0)#torch.Size([26?, 3, 224, 224])
            image_features = torch.stack(gc_eval, dim=0).float() #+ fg_features
            fg_text_features_fn = torch.stack(fg_text_eval, dim=0).float()# + fg_text_features1#torch.Size([20, 3, 224, 224])

            
            # 对比损失的标签（0表示不同，1表示相同），这里假设所有类别对相同，标签为1
            contrastive_labels = torch.ones(clip_features_fg1.size(0)).to(device)
            # 初始化对比损失函数
            contrastive_loss = ContrastiveLoss(margin=1.0)
            # 计算损失
            L_con = contrastive_loss(clip_features_fg1, clip_features_bg1, contrastive_labels)

            L_OTM2 = recam_predictor(image_features,label)#0.6-0.598

            loss_fn = SemanticLoss(fg_text_features_fn ,label)
            loss_ce = F.multilabel_soft_margin_loss(logist, label)
            
            L_OTM = OTMLoss(clip_forward(clip_model, fg_224_eval, fg_label[fg_indices], dname='voc'), 1)

            L_BTM = BTMLoss(clip_forward(clip_model, bg_224_eval, fg_label[fg_indices], dname='voc'), 1)

            L_CBS = CBSLoss(clip_model, fg_224_eval)

            L_REG = torch.mean(x)

            #loss = hyper[0] * L_OTM + hyper[1] * L_BTM + hyper[2] * L_CBS + hyper[3] * L_REG
            loss = hyper[0] * L_OTM + hyper[1] * L_BTM + hyper[2] * L_CBS + hyper[3] * L_REG + 0.05*loss_ce +0.6*L_OTM2 + 0.4*loss_fn + L_con #+0.002*loss_bg
            #loss = hyper[0] * L_OTM + hyper[1] * L_BTM + hyper[2] * L_CBS + hyper[3] * L_REG  +0.6*L_OTM2 + 0.4*loss_fn + L_con
            #loss = hyper[0] * L_OTM + hyper[1] * L_BTM + hyper[2] * L_CBS + hyper[3] * L_REG + L_OTM2 + bg_L_OTM2

            
            loss.backward()
            optimizer.step()

            avg_meter.add({'loss1': loss.item(), 'L_OTM2': L_OTM2.item(), 'L_BTM': L_BTM.item(), 'L_CBS': L_CBS.item(), 'loss_fn': loss_fn.item(), 'L_con': L_con.item(), #'loss_bg': loss_bg.item(),
                           'L_REG': L_REG.item()})

            if (optimizer.global_step - 1) % 200 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'L_OTM2:%.4f' % (avg_meter.pop('L_OTM2')),
                      'loss_fn:%.4f' % (avg_meter.pop('loss_fn')),
                      'L_con:%.4f' % (avg_meter.pop('L_con')),
                      #'loss_bg:%.4f' % (avg_meter.pop('loss_bg')),
                      #'L_regu:%.4f' % (avg_meter.pop('L_regu')),
                      'L_BTM:%.4f' % (avg_meter.pop('L_BTM')),
                      'L_CBS:%.4f' % (avg_meter.pop('L_CBS')),
                      'L_REG:%.4f' % (avg_meter.pop('L_REG')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

                # visualize class activation maps during training if needed.
                # visual_debug(img, label, x, 'vis/clims_v2_voc12_cam_vis', optimizer.global_step, num_classes=21,
                #             dataset='coco', phase='train')

        # validate(model, val_data_loader)
        timer.reset_stage()
    end_time = time.time()
    print("##########end_time:",end_time)
    train_time = end_time - start_time
    print("##########train_time:",train_time)
    torch.save(model.module.state_dict(), args.clims_weights_name + '.pth')
    torch.cuda.empty_cache()
