import json
import os
import argparse
import random
import sys
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import load_prototypes
import losses

import models
from datasets.load_dataset import load_dataset
from losses import SampleMarginLoss


def train_one_epoch(model, criterion, prototypes, optimizer, train_loader, epoch, device, args):
    model.train()
    running_loss = 0.0
    loss_function = criterion
    sample_margin = SampleMarginLoss()
    total = 0
    correct = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for batch_index, (images, labels) in enumerate(train_bar):
        images = images.to(device)
        labels = labels.to(device)
        total += len(images)
        optimizer.zero_grad()
        embedding = model.get_body(images)
        weight = model.get_weight()
        if args.approach == 'sce' or args.approach == 'sce_rsm':
            outputs = model.fc(embedding)
        elif 'msce' in args.approach:
            if 'LearningStrategy' in args.approach:
                dataset_mapping = {  # 策略里面，不同数据集用不同最大度数
                    'mnist': 0,
                    'cifar10': 1,
                    'cifar100': 2
                }
                outputs = model.classifier(embedding, labels, model.training, dataset_mapping[args.dataset])
            elif 'virtual' in args.approach:
                outputs = model.classifier(embedding, labels, model.training)
            else:
                outputs = model.classifier(embedding)
                # outputs = torch.mm(embedding, prototypes)
        elif args.approach in ['virtual_softmax', 'virtual_softmax_rsm', 'virtual_focal', 'resultant_virtual',
                               'virtual_learning_strategy', 'virtual_learning_strategy_addfc']:
            outputs = model.fc(embedding, labels, model.training)
        elif args.approach in ['largest_virtual']:
            outputs = model.fc(embedding, labels, model.training, args.select)
        weight_norm = F.normalize(weight, dim=1)
        embedding_norm = F.normalize(embedding, dim=1)
        output_norm = F.linear(embedding_norm, weight_norm)
        sm = sample_margin(output_norm, labels)
        if 'rsm' in args.approach:
            reg = sm
        else:
            reg = 0
        loss = loss_function(outputs, labels) + 0.5 * reg
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum()
        train_bar.desc = "train epoch[{}/{}], loss:{:.7f}".format(epoch + 1, args.epochs, loss)
    return running_loss / len(train_loader), correct / len(train_loader.dataset)


def test(model, criterion, test_loader, approach, prototypes, epoch, tb_writer, device, args):
    model.eval()
    running_loss = 0.0
    loss_function = criterion
    sample_margin = SampleMarginLoss()
    correct = 0.0
    if 'msce' not in approach:
        prototypes = model.fc.weight
        prototypes = prototypes.t()
    class_angles = [[] for _ in range(prototypes.shape[1])]     # 每个类所有样本与中心向量的角度
    class_avg_angles = []       # 每个类的平均角度
    class_max_angles = []       # 每个类最差的那个样本的角度

    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for (images, labels) in test_bar:
            images = images.to(device)
            labels = labels.to(device)
            embedding = model.get_body(images)
            outputs_original = embedding.clone()    # 原始的 logits
            weight = model.get_weight()
            if args.approach == 'sce' or args.approach == 'sce_rsm':
                outputs = model.fc(embedding)
            elif 'msce' in args.approach:
                if 'virtual' in args.approach:
                    outputs = model.classifier(embedding, labels, model.training)
                else:
                    outputs = model.classifier(embedding)
                    # outputs = torch.mm(embedding, prototypes)
            elif args.approach in ['virtual_softmax', 'virtual_softmax_rsm', 'virtual_focal', 'resultant_virtual',
                                   'virtual_learning_strategy', 'virtual_learning_strategy_addfc', 'largest_virtual']:
                outputs = model.fc(embedding, labels, model.training)
            embedding_norm = F.normalize(embedding)
            weight_norm = F.normalize(weight)
            output_norm = F.linear(embedding_norm, weight_norm)
            sm = sample_margin(output_norm, labels)
            if 'RSM' in approach:
                reg = sm
            else:
                reg = 0
            val_loss = loss_function(outputs, labels) + 0.5 * reg
            running_loss += val_loss.item()
            _, pred = outputs.max(1)  # _:返回outputs里的最大值，pred:返回outputs中最大值所在的位置
            correct += pred.eq(labels).sum()
            # 统计每个类的样本与其各自prototype之间的夹角
            if epoch + 1 >= args.epochs:  # 只在最后一次验证时进行统计
                selected_prototypes = prototypes[:, labels]
                selected_prototypes = selected_prototypes.t()
                cosine_similarity = F.cosine_similarity(outputs_original, selected_prototypes, dim=1)
                angles = torch.rad2deg(torch.acos(cosine_similarity))
                for i in range(len(labels)):  # 遍历batch中的每个样本
                    label = labels[i]
                    angle = angles[i]
                    class_angles[label].append(angle.item())
            test_bar.desc = "test(valid) epoch[{}/{}]".format(epoch + 1, args.epochs)

    if epoch + 1 >= args.epochs:
        # disable by huang on 17-6-2024, because error report
        '''
        for i in range(prototypes.shape[1]):
            tb_writer.add_histogram(f'class{i}_angles', torch.tensor(class_angles[i]), args.epochs)
            tb_writer.add_scalar(f"class_avg_angle_{i}", sum(class_angles[i]) / len(class_angles[i]))
            class_avg_angles.append(round(sum(class_angles[i]) / len(class_angles[i]), 2))
        print(class_avg_angles)
        for sublist in class_angles:
            max_angle = max(sublist)
            class_max_angles.append(max_angle)
        class_max_angles = [float("{:.2f}".format(value)) for value in class_max_angles]
        mean_average_angles = sum(class_avg_angles) / len(class_avg_angles)
        print("{:.4f}".format(mean_average_angles))
        # 保存角度数据到文件
        with open(os.path.join(args.dir, 'angles' + str(args.loop_num) + '.txt'), 'w') as f:
            f.write(json.dumps(class_angles))
        with open(os.path.join(args.dir, 'avg_angles.txt'), 'a') as f:
            f.write(json.dumps(class_avg_angles))
            f.write("\n")
        with open(os.path.join(args.dir, 'max_angles.txt'), 'a') as f:
            f.write(json.dumps(class_max_angles))
            f.write("\n")
            mean_class_max_angles = sum(class_max_angles) / len(class_max_angles)
            print("mean_class_max_angles", mean_class_max_angles)
            f.write(json.dumps(round(mean_class_max_angles, 2)))
            f.write("\n")
        with open(os.path.join(args.dir, 'mean_avg_angles.txt'), 'a') as f:
            f.write(json.dumps(round(mean_average_angles, 2)))
            f.write("\n")
        with open(os.path.join(args.dir, 'accuracy.txt'), 'a') as f:
            val_acc = correct / len(test_loader.dataset)
            f.write("{:.2f}%".format(100. * val_acc.item()) + '\n')
        '''
    # return: 验证集平均loss, 预测正确的比例
    return running_loss / len(test_loader), correct / len(test_loader.dataset)


def main(args):
    print(args)
    log_dir = "_".join([re.sub('[./\*&:]', '', str(getattr(args, attr))) for attr in vars(args)])
    # log_dir = "_".join([str(getattr(args, attr)).replace('/', '') for attr in vars(args)])
    print(log_dir)
    log_dir = os.path.join(args.dir, log_dir)
    tb_writer = SummaryWriter(log_dir=log_dir)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_workers = 8
    print('Using {} dataloader workers every process'.format(num_workers))
    train_loader, test_loader = load_dataset(dataset_name=args.dataset, data_root=args.data_root,
                                             batch_size=args.batch_size, num_workers=num_workers)
    prototypes = None
    test_prototypes = None
    dims = None
    if 'msce' not in args.approach:  #  == "sce":
        if args.dataset == "CUB200-2011":
            dims = 200
        elif args.dataset == "cifar100":
            dims = 100
        else:
            dims = 10
    elif 'msce' in args.approach:
        prototypes, test_prototypes, dims = load_prototypes.load_prototypes(args, device)
        print(test_prototypes.shape)

    if args.network == "cnn":
        inc = 1 if args.dataset == 'MNIST' or args.dataset == 'mnist' else 3
        model = models.cnn(dims=dims, prototypes=prototypes, mode=args.approach, scale=args.s, init_weights=True, in_channel=inc).to(device)
    else:
        model = models.__dict__[args.network](dims=dims, prototypes=prototypes, mode=args.approach, scale=args.scale, init_weights=True).to(device) # scale=args.s edited by huang 17-06-2024
    # Get optimizer and optional scheduler.
    if args.dataset == 'MNIST' or args.dataset == 'mnist':
        weight_decay = 1e-4
    else:
        weight_decay = 5e-4
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=weight_decay)
    if args.network == "alexnet" or args.network == "lenet" or args.network == "cnn":
        train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.001) # eta_min=0.0 edited by huang 17-6-2024

    # added by huang 04-22-24
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        print("continue training from checkpoint {}".format(args.resume))

    # Open logger.
    do_log = False
    if args.log_root != "":
        logger = open(args.log_root, "a")
        do_log = True

    # creating a checkpoints folder
    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    if args.loss_type == 'CE':
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.loss_type == 'Focal':
        criterion = losses.FocalLoss(gamma=1).to(device)
    elif args.loss_type == 'LMSoftmax':
        criterion = losses.LMSoftmaxLoss(scale=64).to(device)   # 我一直默认用10
    else:
        print('The loss type is not listed!')
        return
    # Perform training and periodic testing.
    for epoch in range(args.epochs):
        # Train
        avg_train_loss, train_acc = train_one_epoch(model, criterion, prototypes, optimizer, train_loader, epoch, device, args)
        train_scheduler.step()
        # Test
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1 or epoch == 0 or epoch >= args.epochs - 30:
            val_loss, val_acc = test(model, criterion, test_loader, args.approach, test_prototypes, epoch, tb_writer,
                                     device, args)     # 大失误，之前传的prototypes，改正后，效果还有待测
            logline = "TEST : [epoch-%d] [approach-%s]" \
                      " [dims-%d] [network-%s] [lr-%.3f] [scale-%.2f]:" \
                      " [avg_running_loss-%.7f] [val_accuracy-%.4f]" % \
                      (epoch, args.approach, dims, args.network, train_scheduler.get_last_lr()[0], args.scale,
                       avg_train_loss, val_acc)
            if do_log:
                logger.write(logline + "\n")
            print(logline)

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], avg_train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        # modified by huang 04-22-24
        # torch.save(model.state_dict(),
        #           'ckpts/' + args.dataset + '_lt_' + '_if' + args.network + '_' + args.approach + '.pt')
        saved_model_file='ckpts/' + args.dataset + '_lt_' + '_if' + args.network + '_' + args.approach + '_' + str(epoch) + '.pt'
        if args.resume:
            saved_model_file = 'ckpts/' + args.dataset + '_lt_resume' + '_if' + args.network + '_' + args.approach + '_' + str(
                epoch) + '.pt'
        torch.save(model.state_dict(),saved_model_file)
        
    if do_log:
        logger.write("\n")
    tb_writer.close()
    print('Finished Training')


if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
    print(model_names)      # 列出所有可用的网络
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="resnet34", type=str, choices=model_names,
                        help='默认 resnet34')
    parser.add_argument("--dataset", default="cifar10", type=str, help='mnist or cifar10 or cifar100 or SVHN')
    # modified by huang 16-6-24  data_root
    parser.add_argument("--data_root", default='./datasets/', help='Dataset directory')
    parser.add_argument("--approach", default="sce", type=str, help='loss type',
                        choices=['msce', 'sce', 'virtual_softmax', 'msce_virtual', 'virtual_softmax_rsm',
                                 'msce_LMSoftmax', 'virtual_focal', 'sce_rsm', 'resultant_virtual',
                                 'msce_resultant_virtual', 'msce_virtual_learning_strategy', 'largest_virtual',
                                 'virtual_learning_strategy', 'virtual_learning_strategy_addfc'])
    parser.add_argument('--loss_type', default="CE", type=str, help='loss type',
                        choices=['CE', 'Focal', 'LMSoftmax'])
    parser.add_argument("--radius", default=1.0, type=float,
                        help='radius of the prototypes, use 1 for alexnet and 0.1 for resnet')
    # parser.add_argument("--batch_size", default=128, type=int, help='batch size of the training')
    parser.add_argument("--batch_size", default=64, type=int, help='batch size of the training')
    parser.add_argument("--learning_rate", default=0.1, type=float, help='learning rate of the algorithm')
    parser.add_argument("--epochs", default=200, type=int, help='number of epochs to train the network')
    parser.add_argument("--log_root", default="./log.txt", type=str, help='location of logfile to be saved')
    parser.add_argument("--seed", default=123, type=int, help='seed for initializing training.')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    # parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--loop_num', default=0, type=int, help='当前循环趟数')
    parser.add_argument('--dir', default="runs", type=str, help='tensoroard路径')
    parser.add_argument('--scale', default=None, type=float, help='feature缩放大小')
    parser.add_argument('--load_p', default=1, type=float, help='使用什么样的prototypes')
    parser.add_argument('--s', default=1, type=float, help='virtual prototypes的缩放')
    parser.add_argument('--select', default=4, type=int, help='largest_virtual 的模式选择')
    # added by huang 22-4-24
    parser.add_argument('--resume', default=None, type=str, help='continue to train, given a checkpoint file')

    args = parser.parse_args()
    main(args)
