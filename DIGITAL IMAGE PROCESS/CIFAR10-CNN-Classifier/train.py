import os
import re
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import paramparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datetime import datetime
from tabulate import tabulate

import utils as utils
from model import Classifier, Params

class CIFAR10RGB(Dataset):
    """
    Load CIFAR10 dataset
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, train=True, transform=None):
        self.dataset = datasets.CIFAR10(root='./data', train=train, download=True)
        
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        
        self.transform = transform
        
        self.images = []
        self.labels = []
        
        for img, label in self.dataset:
            img_tensor = self.transform(img)
            
            self.images.append(img_tensor)
            self.labels.append(label)
        
        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels)
        
        self.probs = torch.zeros((len(self.labels), len(self.class_names)), dtype=torch.float32)
        self.probs[torch.arange(len(self.labels)), self.labels] = 1.0

        self.n_images = len(self.labels)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        assert idx < self.n_images, f"Invalid idx: {idx} for n_images: {self.n_images}"

        images = self.images[idx, ...]
        probs = self.probs[idx, ...]
        return images, probs



def train_single_epoch(classifier, dataloader, criterion, optimizer, epoch, device):
    """

    :param Classifier classifier:
    :param dataloader:
    :param criterion:
    :param optimizer:
    :param int epoch:
    :param device:
    :return:
    """
    total_loss = 0
    train_total = 0
    train_correct = 0
    n_batches = 0

    # set CNN to training mode
    classifier.train()

    for batch_idx, (inputs, probs) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f'training epoch {epoch}'):
        inputs = inputs.to(device)
        probs = probs.to(device)

        optimizer.zero_grad()
        outputs = classifier(inputs)
        loss = criterion(outputs, probs)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        _, targets = probs.max(1)
        train_total += probs.size(0)
        train_correct += predicted.eq(targets).sum().item()

        n_batches += 1

    mean_loss = total_loss / n_batches

    train_acc = 100. * train_correct / train_total

    return mean_loss, train_acc



def evaluate(classifier, dataloader, criterion, writer, epoch, tb_samples, eval_type, show, save, vis_tb, device):
    total_loss = 0
    total_images = 0
    correct_images = 0

    pause = 1
    vis = show or save or vis_tb

    # set CNN to evaluation mode
    classifier.eval()

    total_test_time = 0
    tb_vis_imgs = []
    tb_batch_ids = None

    import cv2

    n_batches = len(dataloader)
    if writer is not None:
        tb_batch_ids = list(range(n_batches))
        if n_batches > tb_samples > 0:
            np.random.shuffle(tb_batch_ids)
            tb_batch_ids = tb_batch_ids[:tb_samples]

    # disable gradients computation
    with torch.no_grad():
        for batch_id, (inputs, probs) in tqdm(
                enumerate(dataloader), total=n_batches, desc=f'{eval_type}'):
            inputs = inputs.to(device)
            probs = probs.to(device)
            start_t = time.time()

            outputs = classifier(inputs)

            end_t = time.time()
            test_time = end_t - start_t
            total_test_time += test_time

            if criterion is not None:
                loss = criterion(outputs, probs)
                total_loss += loss.item()

            # get predictions and ground truth labels
            _, predicted = outputs.max(1)
            _, targets = probs.max(1)
            
            total_images += probs.size(0)
            is_correct = predicted.eq(targets)
            correct_images += is_correct.sum().item()

            if not vis:
                continue

            # visualization on tensorboard
            vis_img = None
            try:
                vis_img, pause = utils.vis_cls(
                    batch_id, inputs, dataloader.batch_size, targets, predicted, is_correct, 
                    CIFAR10RGB.class_names,  
                    show=show, pause=pause, save=save, save_dir='vis/test_cls')
            except Exception as e:
                print(f"Visualization error: {e}")

            if vis_tb and writer is not None and batch_id in tb_batch_ids and vis_img is not None:
                tb_vis_imgs.append(vis_img)

        if vis_tb and writer is not None and tb_vis_imgs:
            try:
                vis_img_tb = np.concatenate(tb_vis_imgs, axis=0)
                vis_img_tb = cv2.cvtColor(vis_img_tb, cv2.COLOR_BGR2RGB)
                """tensorboard expects channels in the first axis"""
                vis_img_tb = np.transpose(vis_img_tb, axes=[2, 0, 1])
                writer.add_image(f'{eval_type}/vis', vis_img_tb, epoch)
            except Exception as e:
                print(f"Tensorboard visualization error: {e}")

    mean_loss = total_loss / n_batches if n_batches > 0 else 0
    acc = 100. * float(correct_images) / float(total_images) if total_images > 0 else 0

    test_speed = float(total_images) / total_test_time if total_test_time > 0 else 0

    return mean_loss, acc, total_images, test_speed


def test_cls(params, classifier, device):
    test_params: Params.Testing = params.test
    show, save = map(int, test_params.vis)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_set = CIFAR10RGB(train=False, transform=test_transform)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=test_params.batch_size,
                                                  num_workers=test_params.n_workers)

    _, test_acc, n_test, test_speed = evaluate(
        classifier, test_dataloader, criterion=None, device=device,
        writer=None, epoch=0, eval_type='test', tb_samples=0,
        show=show, save=save, vis_tb=0)

    return test_acc, test_speed


def get_dataloaders(train_params, val_params):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = CIFAR10RGB(train=True, transform=train_transform)
    
    num_train = len(train_set)
    indices = list(range(num_train))

    np.random.shuffle(indices)

    assert val_params.ratio > 0, "Zero validation ratio is not allowed "
    split = int(np.floor((1.0 - val_params.ratio) * num_train))

    train_idx, val_idx = indices[:split], indices[split:]

    n_train = len(train_idx)
    n_val = len(val_idx)

    print(f'Training samples: {n_train:d}\n'
          f'Validation samples: {n_val:d}\n')
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_params.batch_size,
                                                   sampler=train_sampler, num_workers=train_params.n_workers)
    val_dataloader = torch.utils.data.DataLoader(train_set, batch_size=val_params.batch_size,
                                                 sampler=val_sampler, num_workers=val_params.n_workers)

    return train_dataloader, val_dataloader



def train_multiple_epochs(params, ckpt, classifier, criterion, optimizer,
                          train_metrics, val_metrics, device):
    """

    :param Params params:
    :param ckpt:
    :param classifier:
    :param criterion:
    :param optimizer:
    :param Metrics train_metrics:
    :param Metrics val_metrics:
    :param device:
    :return:
    """
    start_epoch = 0

    train_dataloader, val_dataloader = get_dataloaders(params.train, params.val)

    if ckpt is not None:
        start_epoch = ckpt['epoch'] + 1

    ckpt_dir = os.path.abspath(os.path.dirname(params.ckpt.path))

    tb_path = os.path.join(ckpt_dir, 'tb')
    if not os.path.isdir(tb_path):
        os.makedirs(tb_path)

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=tb_path)

    val_params: Params.Validation = params.val
    vis_tb, show, save = map(int, val_params.vis)

    print(f'Saving tensorboard summary to: {tb_path}')
    for epoch in range(start_epoch, params.train.n_epochs):

        train_metrics.loss, train_metrics.acc = train_single_epoch(
            classifier, train_dataloader, criterion, optimizer, epoch, device)

        save_ckpt = train_metrics.update(epoch, params.ckpt.save_criteria)

        # write training data for tensorboard
        train_metrics.to_writer(writer)

        if epoch % params.val.gap == 0:
            val_metrics.loss, val_metrics.acc, _, val_speed = evaluate(
                classifier, val_dataloader, criterion,
                writer, epoch, params.val.tb_samples, eval_type='val',
                show=show, save=save, vis_tb=vis_tb, device=device)

            print(f'validation speed: {val_speed:.4f} images / sec')

            save_ckpt_val = val_metrics.update(epoch, params.ckpt.save_criteria)

            save_ckpt = save_ckpt or save_ckpt_val

            # write validation data for tensorboard
            val_metrics.to_writer(writer)

            rows = ('train', 'val')
            cols = ('loss', 'acc', 'min_loss (epoch)', 'max_acc (epoch)')

            status_df = pd.DataFrame(
                np.zeros((len(rows), len(cols)), dtype=object),
                index=rows, columns=cols)

            train_metrics.to_df(status_df)
            val_metrics.to_df(status_df)

            print(f'Epoch: {epoch}')
            print(tabulate(status_df, headers='keys', tablefmt="orgtbl", floatfmt='.3f'))

        # save checkpoint.
        if save_ckpt:
            model_dict = {
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'timestamp': datetime.now().strftime("%y/%m/%d %H:%M:%S.%f"),
            }
            train_metrics.to_dict(model_dict)
            val_metrics.to_dict(model_dict)
            ckpt_path = f'{params.ckpt.path:s}.{epoch:d}'
            print(f'Saving checkpoint to {ckpt_path}')
            torch.save(model_dict, ckpt_path)

def load_ckpt(params, classifier, optimizer, train_metrics, val_metrics, device):
    """

    :param Params.Checkpoint params:
    :param classifier:
    :param optimizer:
    :param Metrics train_metrics:
    :param Metrics val_metrics:
    :param torch.device device:
    :return:
    """
    ckpt_dir = os.path.abspath(os.path.dirname(params.path))
    ckpt_name = os.path.basename(params.path)

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    if not params.load:
        return None

    matching_ckpts = [k for k in os.listdir(ckpt_dir) if
                      os.path.isfile(os.path.join(ckpt_dir, k)) and
                      k.startswith(ckpt_name)]
    if not matching_ckpts:
        msg = f'No checkpoints found matching {ckpt_name} in {ckpt_dir}'
        if params.load == 2:
            raise IOError(msg)
        print(msg)
        return None
    matching_ckpts.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

    ckpt_path = os.path.join(ckpt_dir, matching_ckpts[-1])

    ckpt = torch.load(ckpt_path, map_location=device)  # load checkpoint

    train_metrics.from_dict(ckpt)
    val_metrics.from_dict(ckpt)

    load_str = (f'Loading weights from: {ckpt_path} with:\n'
                f'\ttimestamp: {ckpt["timestamp"]}\n')

    load_str += train_metrics.to_str(epoch=True)
    load_str += val_metrics.to_str()

    print(load_str)

    classifier.load_state_dict(ckpt['classifier'])
    optimizer.load_state_dict(ckpt['optimizer'])

    return ckpt

def init_cls(params, device):
    """

    :param Params params:
    :param torch.device device:
    :return:
    """
    # create modules
    classifier = Classifier().to(device)

    assert isinstance(classifier, nn.Module), 'classifier must be an instance of nn.Module'

    classifier.init_weights()

    # create loss
    criterion = torch.nn.CrossEntropyLoss().to(device)

    parameters = classifier.parameters()

    # create optimizer
    if params.optim.type == 'sgd':
        optimizer = torch.optim.SGD(parameters,
                                    lr=params.optim.lr,
                                    momentum=params.optim.momentum,
                                    weight_decay=params.optim.weight_decay)
    elif params.optim.type == 'adam':
        optimizer = torch.optim.Adam(parameters,
                                     lr=params.optim.lr,
                                     weight_decay=params.optim.weight_decay,
                                     eps=params.optim.eps,
                                     )
    else:
        raise IOError('Invalid optim type: {}'.format(params.optim.type))

    # create metrics
    train_metrics = utils.Metrics('train')
    val_metrics = utils.Metrics('val')

    return classifier, optimizer, criterion, train_metrics, val_metrics

def calculate_marks(accuracy):
    if accuracy < 75.0:
        return 0.0
    elif accuracy >= 85.0:
        return 100.0
    else:
        # Linear scaling from 50% to 100% for accuracy between 75% and 85%
        return 50.0 + (100.0 - 50.0) * (accuracy - 75.0) / (85.0 - 75.0)

def main():
    params: Params = paramparse.process(Params)
    params.process()

    device = utils.get_device(params.use_gpu)

    classifier, optimizer, criterion, train_metrics, val_metrics = init_cls(params, device)

    ckpt = load_ckpt(params.ckpt, classifier, optimizer, train_metrics, val_metrics, device)

    if params.ckpt.load != 2:
        train_multiple_epochs(params, ckpt, classifier, criterion, optimizer,
                              train_metrics, val_metrics, device)

    with torch.no_grad():
        cls_acc, cls_speed = test_cls(params, classifier, device)
        print(f'CIFAR-10 classification accuracy: {cls_acc:.4f}%')
        print(f'classification speed: {cls_speed:.4f} images / sec')
        
        # Calculate and display marks
        marks = calculate_marks(cls_acc)
        print(f'Marks received: {marks:.2f}%')


if __name__ == '__main__':
    main()