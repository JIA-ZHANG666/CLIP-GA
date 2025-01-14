import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from tqdm import tqdm
import numpy as np
import importlib
import os
import imageio
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_gaussian, create_pairwise_bilateral, unary_from_labels
from skimage.segmentation import slic
from skimage.morphology import remove_small_objects, remove_small_holes

import voc12.dataloader
from misc import torchutils, indexing
from PIL import Image

import time
import tracemalloc

start_time = 0
end_time = 0
train_time = 0

cudnn.enabled = True
palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
           64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
           0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
           64,64,0,  192,64,0,  64,192,0, 192,192,0]


def _work(process_id, model, dataset, args):

    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(tqdm(data_loader, position=process_id, desc=f'[PID{process_id}]')):
            img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            orig_img_size = np.asarray(pack['size'])

            # Since pack['img'][0] has shape [2, 3, 281, 500], we need to handle this accordingly
            img = pack['img'][0][0].numpy().transpose(1, 2, 0)  # Taking the first image of the two

            edge, dp = model(pack['img'][0].cuda(non_blocking=True))

            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

            cams = cam_dict['cam']
            #print("##########cam:",cams.shape)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = cams.cuda()
           
            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)
            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()
            rw_pred = keys[rw_pred]

            superpixels = slic(img, n_segments=3000, compactness=20, sigma=2)
            for sp in np.unique(superpixels):
                mask = superpixels == sp
                rw_pred[mask] = np.argmax(np.bincount(rw_pred[mask]))

            crf_pred = rw_pred
            
            crf_pred_color = Image.fromarray(crf_pred.astype(np.uint8), mode='P')
            crf_pred_color.putpalette(palette)
            crf_pred_color.save(os.path.join(args.sem_seg_out_dir_color, img_name + '.png'))

            imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), crf_pred.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')

def run(args):
    start_time = time.time()
    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
    model.eval()

    n_gpus = torch.cuda.device_count()
    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list,
                                                             voc12_root=args.voc12_root,
                                                             scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")

    end_time = time.time()
    
    train_time = end_time - start_time
    torch.cuda.empty_cache()
