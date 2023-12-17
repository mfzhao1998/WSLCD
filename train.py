import cv2
import time
import torch
import torch.nn as nn
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import os
from net import model
import torch.optim as optim
from utils import  Logger, AverageMeter, accuracy
from progress.bar import Bar
from datasets.CD_dataset import  LoadDatasetFromFolder
import numpy as np
from pytorch_lightning import seed_everything

parser = argparse.ArgumentParser(description='PyTorch change detection (weakly supervised)')
parser.add_argument( '--model', default='cam', type=str)
parser.add_argument( '--backbone', default='resnet18', type=str)
parser.add_argument( '--dataset', default='CLCD256', type=str)
parser.add_argument( '--batchsize', default=64, type=int)
parser.add_argument( '--epoch', default=30, type=int)
parser.add_argument('--gpu_id', default='0,1', type=str)
parser.add_argument('--workers', default=6, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--schedule', type=int, nargs='+', default=[15])
parser.add_argument( '--accpath', default='result', type=str)
parser.add_argument( '--pklpath', default='E:/results/weakly_cd', type=str)
parser.add_argument( '--imgsize', default=256, type=int)
parser.add_argument( '--out_stride', default=8, type=int)
#baseline:cam; Equivariant regularization:er; Mutual learning and equivariant regularization:MLER
#Prototype based contrastive learning: pc (pc_intra,pc_cross)
parser.add_argument( '--mode', default='mlr', type=str)  
parser.add_argument( '--th', default=0.15, type=float)#(1-threshold value in MLER)
parser.add_argument( '--ema', default=0, type=float)#ema weight 
parser.add_argument( '--weight', default=0.15, type=float)#background threshold
parser.add_argument( '--wc', default=0.15, type=float)#weighting factor in bce loss
parser.add_argument( '--multiscale', default='multiscale', type=str)
parser.add_argument( '--weightcls', default='weightcls', type=str)
args = parser.parse_args()
seed_everything(42)
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
ema_updater=EMA(args.ema)
class BCELoss_class_weighted(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight # The weight of positive and negative samples in binary classification, with the first term being negative class weight and the second term being positive class weight
    def forward(self, input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - self.weight[1] * target * torch.log(input) - (1 - target) * self.weight[0] * torch.log(1 - input)
        return torch.mean(bce)
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
            p = (p-min_v-e)/(max_v-min_v+e)
            p[p<0] = 0
    return p

def train(train_loader, model, criterion, optimizer, use_cuda,epoch,old_p1,old_p2):
    # switch to train mode
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (hr1_img, hr2_img, cl_label, seg_label, image_name) in enumerate(train_loader):
        # measure data loading time
        scale_factor = 0.3
        data_time.update(time.time() - end)
        x1, x2, targets= hr1_img.cuda(), hr2_img.cuda(), cl_label.float().cuda()

        x1_s = F.interpolate(x1,scale_factor=scale_factor,mode='bilinear',align_corners=True)
        x2_s = F.interpolate(x2,scale_factor=scale_factor,mode='bilinear',align_corners=True)
        N,C,H,W = x1.size()
        cam11, f_proj1 = model(x1,x2)
        label1 = F.adaptive_avg_pool2d(cam11, (1,1))
        sig=nn.Sigmoid()
        label1=sig(label1).squeeze()
        

        label11=torch.zeros([label1.size(0),2])
        label11[label1>=0.5,1]=1
        label11[label1<0.5,0]=1

        cam1 = F.interpolate(max_norm(cam11),scale_factor=scale_factor,mode='bilinear',align_corners=True)

        cam2, f_proj2 = model(x1_s,x2_s)    
        label2 = F.adaptive_avg_pool2d(cam2, (1,1))
        label2=sig(label2).squeeze()
        cam2=max_norm(cam2)

        label22=torch.zeros([label2.size(0),2])
        label22[label2>=0.5,1]=1
        label22[label2<0.5,0]=1


        loss_cls1 = criterion(label1, targets)
        loss_cls2 = criterion(label2, targets)

        ns,cs,hs,ws = cam2.size()
        cam1_clone=cam1.clone()
        cam2_clone=cam2.clone()
        cam1_clone[torch.where(cam1_clone<args.weight)]=0
        cam2_clone[torch.where(cam1_clone<args.weight)]=0
        cam1_clone[torch.where(cam2_clone<args.weight)]=0
        cam2_clone[torch.where(cam2_clone<args.weight)]=0
        if args.mode=='er':
            loss_er=torch.abs(cam1_clone-cam2_clone)
            loss_er = torch.mean(loss_er)
        elif args.mode=='mlr':
            cam1_copy=cam1_clone.clone().detach()
            cam2_copy=cam2_clone.clone().detach()
            cam1_copy[torch.where((cam1_copy>(1-args.th)))]=1
           
            cam2_copy[torch.where((cam2_copy>(1-args.th)))]=1
            loss_er1 = torch.abs(cam1_copy-cam2_clone)
            loss_er2 = torch.abs(cam1_clone-cam2_copy)
            loss_er = torch.mean(loss_er1)+torch.mean(loss_er2)
      
        cl_label1 = torch.sparse.torch.eye(2)
        cl_label1=cl_label1.index_select(0,cl_label)
        cl_label1=cl_label1.cuda()
        a=torch.where(cl_label1[:,1]==1)
        cl_label1[a[0],0]=1

        f_proj1 = F.interpolate(f_proj1, size=f_proj2.size()[-1],mode='bilinear',align_corners=True)
        cam1 = F.interpolate(cam1, size=f_proj2.size()[-1],mode='bilinear',align_corners=True)
        cam2 = F.interpolate(cam2, size=f_proj2.size()[-1],mode='bilinear',align_corners=True)

        with torch.no_grad():
            # source
            fea1 = f_proj1.detach()
            c_fea1 = fea1.shape[1]
            cam1= F.relu(cam1.detach())
            cam1_0=1-cam1
            cam1=torch.cat([cam1_0,cam1],dim=1)
            cam1=cam1*cl_label1.unsqueeze(2).unsqueeze(3)
            scores1=cam1*cl_label1.unsqueeze(2).unsqueeze(3)
            pseudo_label1 = scores1.argmax(dim=1, keepdim=True)
            n_sc1, c_sc1, h_sc1, w_sc1 = scores1.shape
            scores1 = scores1.transpose(0, 1)
            fea1 = fea1.permute(0, 2, 3, 1).reshape(-1, c_fea1)
            
            top_values, top_indices = torch.topk(cam1.transpose(0, 1).reshape(c_sc1, -1),
                                                    k=h_sc1 * w_sc1 // 8, dim=-1)
            prototypes1 = torch.zeros(c_sc1, c_fea1).cuda()  # [2, 128]
            for i in range(c_sc1):
                top_fea = fea1[top_indices[i]]
                prototypes1[i] = torch.sum(top_values[i].unsqueeze(-1) * top_fea, dim=0) /  (torch.sum(top_values[i])+1e-5)

            # target
            fea2 = f_proj2.detach()
            c_fea2 = fea2.shape[1]

            cam2= F.relu(cam2.detach())
            cam2_0=1-cam2
            cam2=torch.cat([cam2_0,cam2],dim=1)
            cam2=cam2*cl_label1.unsqueeze(2).unsqueeze(3)
            scores2 = cam2*cl_label1.unsqueeze(2).unsqueeze(3)
            pseudo_label2 = scores2.argmax(dim=1, keepdim=True)

            n_sc2, c_sc2, h_sc2, w_sc2 = scores2.shape
            scores2 = scores2.transpose(0, 1)
            fea2 = fea2.permute(0, 2, 3, 1).reshape(-1, c_fea2)
            top_values2, top_indices2 = torch.topk(cam2.transpose(0, 1).reshape(c_sc2, -1),
                                                    k=h_sc2 * w_sc2 // 8, dim=-1)
            prototypes2 = torch.zeros(c_sc2, c_fea2).cuda()
            for i in range(c_sc2):
                top_fea2 = fea2[top_indices2[i]]
                prototypes2[i] = torch.sum(top_values2[i].unsqueeze(-1) * top_fea2, dim=0) / (torch.sum(top_values2[i])+1e-5)

            if args.ema:
                if epoch==0 and batch_idx==0:
                    prototypes1=prototypes1
                    prototypes2=prototypes2
                    old_p1=prototypes1.clone()
                    old_p2=prototypes2.clone()
                else:
                    prototypes1=ema_updater.update_average(old_p1,prototypes1)
                    prototypes2=ema_updater.update_average(old_p2,prototypes2)
                    old_p1=prototypes1.clone()
                    old_p2=prototypes2.clone()

            # L2 Norm
            prototypes2 = F.normalize(prototypes2, dim=-1)
            prototypes1 = F.normalize(prototypes1, dim=-1)

        # intra_view
        # for source
        n_f, c_f, h_f, w_f = f_proj1.shape
        f_proj1 = f_proj1.permute(0, 2, 3, 1).reshape(n_f * h_f * w_f, c_f)
        f_proj1 = F.normalize(f_proj1, dim=-1)
        pseudo_label1 = pseudo_label1.reshape(-1)
        positives1 = prototypes1[pseudo_label1]
        negitives1 = prototypes1

        # for target
        n_f, c_f, h_f, w_f = f_proj2.shape
        f_proj2 = f_proj2.permute(0, 2, 3, 1).reshape(n_f * h_f * w_f, c_f)
        f_proj2 = F.normalize(f_proj2, dim=-1)
        pseudo_label2 = pseudo_label2.reshape(-1)
        positives2 = prototypes2[pseudo_label2]
        negitives2 = prototypes2

        A1 = torch.exp(torch.sum(f_proj1 * positives1, dim=-1) / 0.1)
        A2 = torch.sum(torch.exp(torch.matmul(f_proj1, negitives1.transpose(0, 1)) / 0.1), dim=-1)
        loss_nce1 = torch.mean(-1 * torch.log(A1 / A2))
        A3 = torch.exp(torch.sum(f_proj2 * positives2, dim=-1) / 0.1)
        A4 = torch.sum(torch.exp(torch.matmul(f_proj2, negitives2.transpose(0, 1)) / 0.1), dim=-1)
        loss_nce2 = torch.mean(-1 * torch.log(A3 / A4))

        loss_cross_nce1 =  (loss_nce1 + loss_nce2) / 2


        #cross-view
        positives1_cos = prototypes2[pseudo_label1]
        negitives1_cos = prototypes2

        # for target
        positives2_cos = prototypes1[pseudo_label2]
        negitives2_cos = prototypes1

        A5 = torch.exp(torch.sum(f_proj1 * positives1_cos, dim=-1) / 0.1)
        A6 = torch.sum(torch.exp(torch.matmul(f_proj1, negitives1_cos.transpose(0, 1)) / 0.1), dim=-1)
        loss_nce3 = torch.mean(-1 * torch.log(A5 / A6))

        A7 = torch.exp(torch.sum(f_proj2 * positives2_cos, dim=-1) / 0.1)
        A8 = torch.sum(torch.exp(torch.matmul(f_proj2, negitives2_cos.transpose(0, 1)) / 0.1), dim=-1)
        loss_nce4 = torch.mean(-1 * torch.log(A7 / A8))

        loss_cross_nce2 =  (loss_nce3 + loss_nce4) / 2

        
        loss_cls = (loss_cls1 + loss_cls2)/2 


        if args.model=='cam':
            loss=loss_cls
        elif args.model=='er':
            loss=loss_cls+loss_er
        elif args.model=='mlr':
            loss=loss_cls+loss_er
        elif args.model=='pc':
            loss=loss_cls+loss_er+(loss_cross_nce1+loss_cross_nce2)/2
        elif args.model=='pc_intra':
            loss=loss_cls+loss_er+loss_cross_nce1
        elif args.model=='pc_cross':
            loss=loss_cls+loss_er+loss_cross_nce2

        if args.ema:
            if epoch>1:
                loss = loss
            else:
                loss = loss_cls +loss_er


        prec1= accuracy(label11.cuda().data, targets.data, topk=(1, ))[0]
        losses.update(loss.item(), x1.size(0))
        top1.update(prec1.item(), x1.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg,old_p1,old_p2)


if __name__ == '__main__':
    if args.model=='cam':
        args.ema=0
        args.weight=0
        args.th=0
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
    use_cuda= True
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
    train1_dir='E:/CD_dataset/'+datasetname+'_0.2/train/A'
    train2_dir='E:/CD_dataset/'+datasetname+'_0.2/train/B'
    label_train='E:/CD_dataset/'+datasetname+'_0.2/train/label'
    train_dataset = LoadDatasetFromFolder(suffix,train1_dir, train2_dir, label_train,imgsize,True)
    train_data_loader = DataLoader(train_dataset, batch_size=batchsize,
                                   shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0001,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=args.schedule,
                                                    gamma=0.1)
    if len(args.gpu_id)>1:
        model=nn.DataParallel(model).cuda()
    else:
        model.cuda()


    weight=[args.wc,1-args.wc]
    if args.weightcls=='noweightcls':
        ce = nn.BCELoss()
    elif args.weightcls=='weightcls':
        ce=BCELoss_class_weighted(weight)

    old_p1=0
    old_p2=0
    for epoch in range (num_epoches): 
        print('\nEpoch: [%d | %d] LR1: %f ' % (epoch + 1, num_epoches, optimizer.param_groups[0]['lr']))
        train_loss, train_acc,old_p1,old_p2 = train(train_data_loader, model,ce, optimizer,use_cuda,epoch,old_p1,old_p2)    
        lr_scheduler.step()
    if len(args.gpu_id)>1:
        torch.save(model.module.state_dict(), os.path.join(args.pklpath, pklname))
    else:
        torch.save(model.state_dict(), os.path.join(args.pklpath, pklname))  
    torch.cuda.empty_cache()
