import cv2
import os
import torch
import os.path as osp
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import distributed

import importlib
from imutils import visual_debug
from clip_utils import clip_forward
from clip_loss import SimMaxLoss, SimMinLoss, BackgroundSuppressionLoss
import mscoco.dataloader
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

new_class_names_coco = ['person with clothes,people,human','bicycle','car','motorbike','aeroplane',
                    'bus','train','truck','boat','traffic light',
                    'fire hydrant','stop sign','parking meter','bench','bird avian',
                    'cat','dog','horse','sheep','cow',
                    'elephant','bear','zebra','giraffe','backpack,bag',
                    'umbrella,parasol','handbag,purse','necktie','suitcase','frisbee',
                    'skis','sknowboard','sports ball','kite','baseball bat',
                    'glove','skateboard','surfboard','tennis racket','bottle',
                    'wine glass','cup','fork','knife','dessertspoon',
                    'bowl','banana','apple','sandwich','orange',
                    'broccoli','carrot','hot dog','pizza','donut',
                    'cake','chair seat','sofa','pottedplant','bed',
                    'diningtable','toilet','tvmonitor screen','laptop','mouse',
                    'remote control','keyboard','cell phone','microwave','oven',
                    'toaster','sink','refrigerator','book','clock',
                    'vase','scissors','teddy bear','hairdrier,blowdrier','toothbrush',
                    ]

with open("./graph/CM_kg_57_info.json","rb") as f:
        info = json.load(f)
        KG_COCO_info = info['KG_COCO_info']
        graph_adj_mat = np.asarray(KG_COCO_info['S'])
        
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

def reduce_mean(tensor, nprocs):
    # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt


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


def run(args):
    model = getattr(importlib.import_module(args.clims_network), 'CLIMS')(n_classes=80)
    
    
            
    model.load_state_dict(torch.load('cam-baseline-coco/res50_cam.pth'), strict=True)

    ################################################
    voc_data = get_voc_data()
    graph_learner = graph.graph_layer.GraphLearner(**voc_data)
    graph_learner = torch.nn.DataParallel(graph_learner).cuda()
    graph_learner.train()
    conv1 = nn.Conv2d(20,3,1,1).cuda()

    SemanticLoss = net.resnet50_clims.SemanticContextLoss(80, 256)
    SemanticLoss = torch.nn.DataParallel(SemanticLoss).cuda()
    SemanticLoss.train()

    recam_predictor = net.resnet50_clims.Class_Predictor(80, 256)
    recam_predictor = torch.nn.DataParallel(recam_predictor).cuda()
    recam_predictor.train()

    graph_fusion = graph.graph_layer.GraphFusion(256, 80)
    graph_fusion = torch.nn.DataParallel(graph_fusion).cuda()
    graph_fusion.train()
    ##########################################
       
    train_dataset = mscoco.dataloader.COCOClassificationDataset(
        image_dir=osp.join(args.mscoco_root, 'train2014/train2014/'),
        anno_path=osp.join(args.mscoco_root, 'annotations/instances_train2014.json'),
        labels_path='./mscoco/train_labels.npy',
        resize_long=(320, 640), hor_flip=True, crop_size=512, crop_method="random")
    
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.clims_num_epoches
    
    
    # val_dataset = mscoco.dataloader.COCOClassificationDataset(
    #     image_dir=osp.join(args.mscoco_root, 'val2014/'),
    #     anno_path=osp.join(args.mscoco_root, 'annotations/instances_val2014.json'),
    #     labels_path='./mscoco/val_labels.npy', crop_size=512)
    # val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size, shuffle=False,
    #                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.clims_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.clims_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': graph_learner.parameters(), 'lr': 0.0001, 'weight_decay': args.cam_weight_decay},#0.0001-0.5933
        #{'params': conv1.parameters(), 'lr': 10 * args.clims_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': recam_predictor.parameters(), 'lr': 0.1, 'weight_decay': args.cam_weight_decay},
        {'params': SemanticLoss.parameters(), 'lr': 0.0001, 'weight_decay': args.cam_weight_decay},#0.4
        {'params': graph_fusion.parameters(), 'lr': 0.0001, 'weight_decay': args.cam_weight_decay},
    ], lr=args.clims_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

  
   
    model = torch.nn.DataParallel(model).cuda()

    model.train()

    # CLIP
    import clip
   
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    if args.clip == 'RN50x4':
        clip_input_size = 288
    else:
        clip_input_size = 224

    # Loss
    hyper = [float(h) for h in args.hyper.split(',')]
    OTMLoss = SimMaxLoss()
    BTMLoss = SimMinLoss()
    CBSLoss = BackgroundSuppressionLoss(threshold=args.cbs_loss_thresh, dname='coco')
    print('clims on coco')
    print(hyper)


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

    fg_text_features1 = zeroshot_classifier(new_class_names_coco, ['a photo of a {}.'], clip_model)#fg_text_features: torch.Size([20, 512])
    #fg_text_features1 = zeroshot_classifier(new_class_names, ['a clean origami {}.'], clip_model)
    fg_text_features = graph_learner(fg_text_features1, graph_adj_mat)#.detach()
    #print("##########fg_text_features1:",fg_text_features.shape)
    
###############################################

    def preprocess(labels):
        new_labels = []
        for n in range(labels.size(0)):
            for idx in range(0, labels.size(1)):
                temp = torch.zeros(1, labels.size(1)).long()
                if labels[n, idx] == 1:
                    temp[0, idx] = 1
                new_labels.append(temp)
        return torch.cat(new_labels, dim=0).cuda()

    for ep in range(args.clims_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.clims_num_epoches))
        

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            img = img.cuda()
            label = pack['label'].cuda(non_blocking=True)

            fg_label = preprocess(label.cpu())
            x,logist = model(img)
            N, _, _, _ = x.size()
            optimizer.zero_grad()

            # foreground indices
            fg_indices = torch.nonzero(label.reshape(-1) == 1, as_tuple=False).squeeze()

            g_cam_224 = F.interpolate(x, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)
            cam_224 = F.interpolate(x, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True).reshape(N * 80, 1, clip_input_size,
                                                                                                clip_input_size)
            img_224 = F.interpolate(img, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)

            fg_224_eval = []
            bg_224_eval = []
            gc_fg_224_eval = []
            gc_bg_224_eval = []
            fg_img_224 = []
            total_loss = 0
            bg_total_loss = 0
            gc_eval = []
            fg_text_eval = []
            temp_idx = torch.nonzero(label == 1, as_tuple=False)
            for j in range(temp_idx.shape[0]):
                fg_224_eval.append(cam_224[fg_indices[j]] * img_224[temp_idx[j, 0]])
                bg_224_eval.append((1 - cam_224[fg_indices[j]]) * img_224[temp_idx[j, 0]])

            ###############################################################################
            for class_idx in range(80):
                class_indices = torch.nonzero(label[:, class_idx] == 1).squeeze()
        
                if class_indices.numel() == 0:
                    # 如果没有图像属于该类，跳过该类
                    gc_fg_224_eval.append(torch.zeros(3, 224, 224).to(device))
                    gc_bg_224_eval.append(torch.zeros(3, 224, 224).to(device))
        
                else:
                    selected_idx = class_indices.item() if class_indices.dim() == 0 else class_indices[0]
        
                    gc_fg_224_eval.append((cam_224[selected_idx * 80 + class_idx] * img_224[selected_idx]).squeeze(0))
                    gc_bg_224_eval.append(((1 - cam_224[selected_idx * 80 + class_idx]) * img_224[selected_idx]).squeeze(0))
    
            gc_fg_224_eval = torch.stack(gc_fg_224_eval, dim=0)
            gc_bg_224_eval = torch.stack(gc_bg_224_eval, dim=0)
            clip_features_fg1 = clip_model.encode_image(gc_fg_224_eval)#.detach()
            #print("########clip_features_fg:",clip_features_fg.shape)
            clip_features_bg1 = clip_model.encode_image(gc_bg_224_eval)#.detach()

            fg_features, bg_features = generate_clip_input(img_224, g_cam_224, label, clip_model, clip_input_size, device)
            for i in range(img.shape[0]):
                
                edges = cosine_similarity(fg_features[i])#0.4*loss_fn-0.5946
                edges = torch.tensor(edges, dtype=torch.float).cuda()
                #edges = edges.type(fg_text_features.dtype)
                
                #edge_index, edge_weight = dense_to_sparse(edges)
                image_features = graph_learner(fg_features[i], edges)
                fg_text_features2, image_features = graph_fusion(fg_text_features, image_features)
                
                gc_eval.append(image_features)
                fg_text_eval.append(fg_text_features2)

            fg_224_eval = torch.stack(fg_224_eval, dim=0)
            bg_224_eval = torch.stack(bg_224_eval, dim=0)
            image_features = torch.stack(gc_eval, dim=0).float() #+ fg_features
            fg_text_features_fn = torch.stack(fg_text_eval, dim=0).float()# + fg_text_features1#torch.Size([20, 3, 224, 224])

             # 对比损失的标签（0表示不同，1表示相同），这里假设所有类别对相同，标签为1
            contrastive_labels = torch.ones(clip_features_fg1.size(0)).to(device)
            # 初始化对比损失函数
            contrastive_loss = ContrastiveLoss(margin=1.0)
            # 计算损失
            L_con = contrastive_loss(clip_features_fg1, clip_features_bg1, contrastive_labels)

            L_OTM2 = recam_predictor(image_features,label)#0.6-0.598

            #loss_bg = torch.relu(bg_features).mean()
            #L_regu = regularization_loss(image_features) + regularization_loss(bg_features)#0.2-0.596

            
            loss_fn = SemanticLoss(fg_text_features_fn ,label.float())
            loss_ce = F.multilabel_soft_margin_loss(logist, label)

            L_OTM = OTMLoss(clip_forward(clip_model, fg_224_eval, fg_label[fg_indices], dname='coco'), 1)

            L_BTM = BTMLoss(clip_forward(clip_model, bg_224_eval, fg_label[fg_indices], dname='coco'), 1)

            L_CBS = CBSLoss(clip_model, fg_224_eval)

            L_REG = torch.mean(x)

            #loss = hyper[0] * L_OTM + hyper[1] * L_BTM + hyper[2] * L_CBS + hyper[3] * L_REG
            loss = hyper[0] * L_OTM + hyper[1] * L_BTM + hyper[2] * L_CBS + hyper[3] * L_REG + 0.05*loss_ce +0.6*L_OTM2 + 0.4*loss_fn + L_con

            loss.backward()
            optimizer.step()

            #if args.use_distributed_train:
                #loss = reduce_mean(loss, distributed.get_world_size())
                #if args.local_rank != 0:
                    #continue

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

                # visual_debug(img, label, x, 'vis/clims_coco_cam_vis', optimizer.global_step, num_classes=81,
                #             dataset='coco', phase='train')

        # validate(model, val_data_loader)
        timer.reset_stage()

    # torch.save(model.module.state_dict(),
    #            args.clims_weights_name + f'{hyper[0]}_{hyper[1]}_{hyper[2]}_{hyper[3]}_K({hyper[4]})_ep({args.clims_num_epoches})_lr({args.clims_learning_rate}).pth')
    # torch.save(model.module.state_dict(), args.clims_weights_name + '.pth')
    # torch.cuda.empty_cache()

   
    torch.save(model.module.state_dict(), args.clims_weights_name + '.pth')
    torch.cuda.empty_cache()
