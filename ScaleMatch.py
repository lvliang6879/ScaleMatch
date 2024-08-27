import os
import time
import yaml
import pprint
import logging
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from evaluate import evaluate
from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus_scalematch import DeepLabV3Plus
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed
import random
import torch.distributed as dist
from util.train_utils import (DictAverageMeter, confidence_weighted_loss,
                              cutmix_img_, cutmix_mask, generate_lambda_schedule)

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=1, type=int)
parser.add_argument('--port', default=None, type=int)


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    # if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_miou(pre, mask, cfg):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    intersection, union, target = \
        intersectionAndUnion(pre.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

    reduced_intersection = torch.from_numpy(intersection).cuda()
    reduced_union = torch.from_numpy(union).cuda()
    reduced_target = torch.from_numpy(target).cuda()
    dist.all_reduce(reduced_intersection)
    dist.all_reduce(reduced_union)
    dist.all_reduce(reduced_target)

    intersection_meter.update(reduced_intersection.cpu().numpy())
    union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    non_zero_iou_class = iou_class[iou_class != 0]
    mIOU = np.mean(iou_class)

    return mIOU, iou_class


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    ddp = True if world_size > 1 else False
    amp = cfg['amp']

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    conf_thresh = cfg['conf_thresh']

    model = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    if ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                          output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    if ddp:
        trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
        trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
        trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
        trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
        valsampler = torch.utils.data.distributed.DistributedSampler(valset)
        valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                               drop_last=False, sampler=valsampler)
    else:
        trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=1, shuffle=True, drop_last=True)
        trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=1, shuffle=True, drop_last=True)
        valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    total_epochs = cfg['epochs']
    total_iters = len(trainloader_u) * total_epochs
    epoch = -1
    previous_best = 0.0
    ETA = 0.0

    scaler = torch.cuda.amp.GradScaler()

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    is_best = False
    for epoch in range(epoch + 1, total_epochs):
        start_time = time.time()


        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}, ETA: {:.2f}M'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best, ETA / 60))

        log_avg = DictAverageMeter()

        if ddp:
            trainloader_l.sampler.set_epoch(epoch)
            trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        current_lambda = generate_lambda_schedule(epoch, total_epochs, total_epochs // 2)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2, mask_u_w_gt),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _, _)) in enumerate(loader):

            t0 = time.time()
            _, _, H, W = img_u_w.shape
       
            random_scale = random.choice(cfg['img_scales'])
            feature_scale = random.choice(cfg['feat_s_scales'] if random_scale > 1 else cfg['feat_l_scales'] )

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            iters = epoch * len(trainloader_u) + i

            # CutMix images
            cutmix_img_(img_u_s1, img_u_s1_mix, cutmix_box1)
            cutmix_img_(img_u_s2, img_u_s2_mix, cutmix_box2)

            # Use AMP
            with torch.cuda.amp.autocast(enabled=amp):

                model.eval()
                if amp:
                    pred_u_w_mix = model(img_u_w_mix, scale_factor=None, scales=None)
                    conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)
                else:
                    with torch.no_grad():
                        pred_u_w_mix = model(img_u_w_mix).detach()
                        conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)

                # Generate previous guidance

                model.train()

                num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

                pred = model(torch.cat((img_x, img_u_w)), scale_factor=random_scale, feature_scale=feature_scale)
                pred_u_s = model(img_u_s1, scale_factor=None, scales=None)

                if epoch < cfg['warm_up']:
                    pred_u_w = pred['pred_ori'][num_lb:]
                else:
                    pred_u_w = pred['pred_joint'][num_lb:]

                pred_u_w = pred_u_w.detach()
                conf_u_w, mask_u_w = pred_u_w.softmax(dim=1).max(dim=1)

                # CutMix labels
                mask_u_w_cutmixed1 = cutmix_mask(mask_u_w, mask_u_w_mix, cutmix_box1)
                # mask_u_w_cutmixed2 = cutmix_mask(mask_u_w, mask_u_w_mix, cutmix_box2)
                conf_u_w_cutmixed1 = cutmix_mask(conf_u_w, conf_u_w_mix, cutmix_box1)
                # conf_u_w_cutmixed2 = cutmix_mask(conf_u_w, conf_u_w_mix, cutmix_box2)
                ignore_mask_cutmixed1 = cutmix_mask(ignore_mask, ignore_mask_mix, cutmix_box1)
                # ignore_mask_cutmixed2 = cutmix_mask(ignore_mask, ignore_mask_mix, cutmix_box2)

                # Generate previous guidance

                pred_x_joint = pred['pred_joint'][:num_lb]

                pred_u_w_scale = pred['pred_size'][num_lb:]
                pred_u_w_fp = pred['pred_fp'][num_lb:]

                loss_x = criterion_l(pred_x_joint, mask_x)

                loss_u_s1 = criterion_u(pred_u_s, mask_u_w_cutmixed1)
                loss_u_s1 = confidence_weighted_loss(loss_u_s1, conf_u_w_cutmixed1, ignore_mask_cutmixed1,
                                                     conf_thresh=conf_thresh)
                loss_u_size = criterion_u(pred_u_w_scale, mask_u_w)
                loss_u_size = confidence_weighted_loss(loss_u_size, conf_u_w, ignore_mask, conf_thresh=conf_thresh)

                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = confidence_weighted_loss(loss_u_w_fp, conf_u_w, ignore_mask, conf_thresh=conf_thresh)

                mask_ratio = ((conf_u_w >= conf_thresh) & (ignore_mask != 255)).sum().item() / \
                             (ignore_mask != 255).sum()

                loss_standard = loss_u_s1 * 0.25 + loss_u_size * 0.25 + loss_u_w_fp * 0.5

                total_loss = (loss_x + loss_standard) / 2.0

            if ddp:
                torch.distributed.barrier()

            optimizer.zero_grad()
            if amp:
                loss = scaler.scale(total_loss)
                loss.backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            # Logging
            log_avg.update({
                'iter time': time.time() - t0,
                'Total loss': total_loss,  # Logging original loss (=total loss), not AMP scaled loss
                'Loss x': loss_x,
                'Loss u_s': loss_u_s1,
                'Loss u_scale': loss_u_size,
                'Loss w_fp_scale': loss_u_w_fp,
                'Mask ratio': mask_ratio,
            })

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                for k, v in log_avg.avgs.items():
                    writer.add_scalar('train/' + k, v, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(f'Iters: {i}, ' + str(log_avg))
                log_avg.reset()

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, ddp)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))

            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))

        end_time = time.time()
        time_per_epoch = end_time - start_time
        ETA = (total_epochs - (epoch + 1)) * time_per_epoch


if __name__ == '__main__':
    main()
