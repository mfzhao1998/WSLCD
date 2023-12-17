
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import csv
import os
from net import model
from progress.bar import Bar
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage import io
from metircs import Metrics
from datasets.CD_dataset import  LoadDatasetFromFolder
import argparse

parser = argparse.ArgumentParser(description='PyTorch change detection (weakly supervised)')
parser.add_argument( '--model', default='cam', type=str)
parser.add_argument( '--backbone', default='resnet18', type=str)
parser.add_argument( '--dataset', default='BCD200', type=str)
parser.add_argument( '--batchsize', default=64, type=int)
parser.add_argument( '--epoch', default=30, type=int)
parser.add_argument('--gpu_id', default='0,1', type=str)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--schedule', type=int, nargs='+', default=[15])
parser.add_argument( '--accpath', default='result', type=str)
parser.add_argument( '--pklpath', default='E:/results/weakly_cd', type=str)
parser.add_argument( '--imgsize', default=256, type=int)
parser.add_argument( '--out_stride', default=8, type=int)
parser.add_argument( '--mode', default='mlr', type=str)
parser.add_argument( '--th', default=0.15, type=float)
parser.add_argument( '--ema', default=0, type=float)
parser.add_argument( '--weight', default=0.15, type=float)
parser.add_argument( '--wc', default=0.15, type=float)
parser.add_argument( '--multiscale', default='multiscale', type=str)
parser.add_argument( '--weightcls', default='weightcls', type=str)
args = parser.parse_args()
def max_norm(p, version='torch', e=1e-7):
    if version is 'torch':
        if p.dim() == 3:
            C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)
            min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C,1,1)
            p = F.relu(p-min_v-e)/(max_v-min_v+e)
        elif p.dim() == 4:
            N, C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
            min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
            p = F.relu(p-min_v-e)/(max_v-min_v+e)
    elif version is 'numpy' or version is 'np':
        if p.ndim == 3:
            C, H, W = p.shape
            p[p<0] = 0
            max_v = np.max(p,(1,2),keepdims=True)
            min_v = np.min(p,(1,2),keepdims=True)
            p = (p-min_v-e)/(max_v-min_v+e)
            p[p<0]=0
        elif p.ndim == 4:
            N, C, H, W = p.shape
            p[p<0] = 0
            max_v = np.max(p,(2,3),keepdims=True)
            min_v = np.min(p,(2,3),keepdims=True)
            p = (p-e)/(max_v-min_v+e)
            p[p<0] = 0
    return p
def get_metrics(norm_cam,seg_label,metrics,cl_label):
    pred=np.zeros((cl_label.size()[0], args.imgsize,args.imgsize))
    th=0.5
    pred[norm_cam[:,0,:,:]>=th]=1  
    pred[torch.where(cl_label==0)[0],:,:]=0
    pred_tensor=torch.from_numpy(pred)
    seg_label = torch.argmax(seg_label, 1).unsqueeze(1)
    seg_label = (seg_label > 0).float()
    metrics=seg_label[:,0,:,:].float(),pred_tensor.float()
    return metrics

if __name__ == '__main__':
    if args.model=='cam':
        args.ema=0
        args.th=0
        args.weight=0
    if args.model=='er' or args.model=='mlr':
        args.mode=args.model
    if args.model=='er' or args.model=='mlr' or args.model=='cam':
        args.ema=0
    # if args.model=='pc' or args.model=='pc_intra' or args.model=='pc_cross':
    #     args.mode='mlr'


    elif args.dataset=='DSIFN256':
        args.imgsize=256
    elif args.dataset=='CLCD256':
        args.imgsize=256
    elif args.dataset=='GCD256':
        args.imgsize=256
    use_cuda=True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = model.Net_sig(output_stride=args.out_stride,backbone=args.backbone,multiscale=args.multiscale)
    datasetname= args.dataset
    suffix=['.png','.jpg','.tif']
    batchsize=args.batchsize
    num_epoches=args.epoch
    num_workers=args.workers
    imgsize=args.imgsize
    pklname=(args.model+"-"+str(args.backbone)+"-"+str(args.out_stride)+"-"+str(args.mode)+"-"+args.multiscale+"-"+args.weightcls+"-"
            +str(args.epoch)+"-"+str(args.th)+"-"+str(args.ema)+"-"+str(args.dataset)+"-"+str(args.imgsize)+"-"+str(args.weight)+"-"+str(args.wc)+".pth")
    accname=(args.model+"-"+str(args.backbone)+"-"+str(args.out_stride)+"-"+str(args.mode)+"-"+args.multiscale+"-"+args.weightcls+"-"
            +str(args.epoch)+"-"+str(args.th)+"-"+str(args.ema)+"-"+str(args.dataset)+"-"+str(args.imgsize)+"-"+str(args.weight)+"-"+str(args.wc)+".csv")
    model.load_state_dict(torch.load(os.path.join(args.pklpath, pklname)))
    model.cuda()
    model.eval()

    train1_dir='E:/CD_dataset/'+datasetname+'_0.2test/train/A'
    train2_dir='E:/CD_dataset/'+datasetname+'_0.2test/train/B'
    label_train='E:/CD_dataset/'+datasetname+'_0.2test/train/label'

    val1_dir='E:/CD_dataset/'+datasetname+'/val/A'
    val2_dir='E:/CD_dataset/'+datasetname+'/val/B'
    label_val='E:/CD_dataset/'+datasetname+'/val/label'

    train_dataset = LoadDatasetFromFolder(suffix, train1_dir, train2_dir, label_train, img_size=args.imgsize, is_train=False)
    train_data_loader = DataLoader(train_dataset, batch_size=batchsize, num_workers=4, pin_memory=True)
    
    bar = Bar('Processing', max=len(train_data_loader))
    metrics_singlescale=Metrics(range(2))
    for batch_idx, (hr1_img, hr2_img, cl_label, seg_label, image_name) in enumerate(train_data_loader):
        bar.suffix  = '({batch}/{size}) Total: {total:}'.format(
                        batch=batch_idx + 1,
                        size=len(train_data_loader),
                        total=bar.elapsed_td,
                        )
        bar.next()
        with torch.no_grad():
            cam, _ = model(hr1_img.cuda(),  hr2_img.cuda())
            cam = F.upsample(cam, args.imgsize, mode='bilinear', align_corners=True)
            cam = cam.cpu().numpy()
        norm_cam_singlescale = max_norm(cam,version='np')
        metrics_singlescale=get_metrics(norm_cam_singlescale,seg_label,metrics_singlescale,cl_label)  
    bar.finish()       
    ciou_singlescale=['ciou']+[str(metrics_singlescale.get_fg_iou())]
    presion_singlescale=['presion']+[str(metrics_singlescale.get_precision())]
    recall_singlescale=['recall']+[str(metrics_singlescale.get_recall())]
    f1_singlescale=['f1']+[str(metrics_singlescale.get_f_score())]
    kappa_singlescale=['kappa']+[str(metrics_singlescale.get_kappa())]
    miou_singlescale=['miou']+[str(metrics_singlescale.get_miou())]
    oa_singlescale=['oa']+[str(metrics_singlescale.get_oa())]

    if os.path.exists(os.path.join(args.accpath, accname)):
        f = open(os.path.join(args.accpath, accname),'a',encoding='utf-8',newline='')
    else:
        f = open(os.path.join(args.accpath, accname),'w',encoding='utf-8',newline='')
    
    csv_writer = csv.writer(f)
    csv_writer.writerow(presion_singlescale)
    csv_writer.writerow(recall_singlescale)
    csv_writer.writerow(f1_singlescale)
    csv_writer.writerow(miou_singlescale)
    csv_writer.writerow(ciou_singlescale)
    csv_writer.writerow(kappa_singlescale)
    csv_writer.writerow(oa_singlescale)
    
    


