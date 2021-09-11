from dataset.CelebA import *
from models.model import DoubleResNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

torch.manual_seed(7777)

import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import argparse

from pathlib import Path
import time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):
    # batch_size = int(args.batch_size / args.ngpus_per_node)
    # num_worker = int(args.num_workers / args.ngpus_per_node)
    now = time.localtime()
    save_path = Path('./') /\
                'DoubleResNetV2' /\
                f'{now.tm_mon}.{now.tm_mday}_{now.tm_hour}:{now.tm_min}:{now.tm_sec}'

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    transforms = T.Compose([
        PairedDataToTensor(),
        PairedDataNormalization(),
        PairedDataResize(512),
        PairedDataRandomHorizontalFlip(prob=.5),
        PairedDataRandomCrop(prob=.5,
                             inter_size=(532, 532),
                             output_size=(512, 512))
    ])
    dataset = CelebADataset(img_path='/home/sin/git/pytorch.segmentation/data/CelebAMask-HQ/CelebA-HQ-img/',
                            seg_path='/home/sin/git/pytorch.segmentation/data/preprocessing-mask/',
                            transforms=transforms)

    train_indicise = [i for i in range(21000)]
    test_indicise = [i for i in range(9000)]
    train_set = Subset(dataset, train_indicise)
    test_set = Subset(dataset, test_indicise)

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              pin_memory=True,
                              num_workers=args.num_workers,
                              shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    net = DoubleResNet(upsample_blocks = [4,4,4,4,4],
                    downsample_blocks = [3,3,3,3],
                    norm_layer = nn.BatchNorm2d)
    net.to(device)
    net.train()

    test_iters = iter(test_loader)
    fixed_img, fixed_label = test_iters.next()
    fixed_img = fixed_img.to(device)
    fixed_img = denormImg(fixed_img)
    fixed_label = fixed_label.to(device)
    fixed_label = denormSeg(fixed_label)

    # loss_function = OhemCELoss(thresh=args.score_thres,
    #                    n_min=args.n_min,
    #                    ignore_lb=args.ignore_idx).to(gpu)
    # loss_function = Cross_entropy2d()

    optim = torch.optim.SGD(
                        net.parameters(),
                        lr=args.learning_rate,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)

    # optim = torch.optim.Adam(g
    #     net.parameters(),
    #     lr=args.learning_rate,
    #     weight_decay=args.weight_decay)
    previous_best_loss = None

    train_iter = iter(train_loader)
    for i in range(args.start_iters, args.num_iters):
        try:
            img, label = next(train_iter)
        except:
            train_iter = iter(train_loader)
            img, label = next(train_iter)

        img = img.to(device)
        label = label.to(device)
        # print(img.shape, label.shape)

        pred = net(img)
        loss = F.cross_entropy(pred, label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss = loss.item()
        print(f"{i}/{args.num_iters}, loss:{loss}")

        with open(save_path / f'loss.log', 'a') as f:
            f.write(f'{i} {loss}\n')
            # if ind % args.verbose_iter == 0:
            # if dist.get_rank()  == 0:
            #     print(f'{len(train_loader)} / {ind}, loss : {loss}')
            # print(dist.get_rank())

        if (i + 1) % args.sample_step == 0:
            with torch.no_grad():
                # x_fake_list = [base_img]
                # x_fake_list.append([base_label])
                # for c_fixed in base_img:
                #     x_fake_list.append(net(base_img))
                pred = net(fixed_img)
                pred = pred.argmax(1)
                pred = denormSeg(pred)

                print(fixed_img.size(), fixed_label.size(), pred.size())
                x_concat = torch.cat((fixed_img, fixed_label, pred), dim=2)
                # sample_path = os.path.join(save_path, '{}-images.jpg'.format(i + 1))
                sample_path = save_path / 'samples' / f'{i+1}-images.jpg'
                if not sample_path.parent.exists():
                    sample_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(x_concat.data.cpu(), sample_path, nrow=args.batch_size, padding=0)
                print('Saved real and fake images into {}...'.format(sample_path))

        if (i + 1) % args.sample_step == 0:
            if previous_best_loss is None:
                previous_best_loss = loss

            if previous_best_loss >= loss:
                previous_best_loss = loss
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                loss_path = save_path / 'models' / f"{i+1}-model:{loss}.pt"
                if not loss_path.parent.exists():
                    loss_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(state, loss_path)
                print(f'Find best loss -> {loss}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', type=int, default='19')
    parser.add_argument('--score_thres', default=0.3 )
    parser.add_argument('--n_min', default=512*512)
    parser.add_argument('--ignore_idx', default=-100)
    parser.add_argument('--learning_rate', default=1e-3)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=5e-4)
    parser.add_argument('--start_iters', default=0, type=int)
    parser.add_argument('--num_iters', default=30000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--save_path', default='/home/sin/save_files/')
    parser.add_argument('--verbose_iter', default=1000)
    parser.add_argument('--sample_step', default=1000)
    args = parser.parse_args()

    # ngpus_per_node = torch.cuda.device_count()
    # print(ngpus_per_node)
    # world_size = ngpus_per_node
    # args.ngpus_per_node = ngpus_per_node
    # mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, ))

    main(args)

    # print(args.n_classes)
    # train(args)
