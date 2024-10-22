import argparse
import json
import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
from sklearn.metrics import f1_score, roc_auc_score


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.deeplab = deeplabv3_resnet101(pretrained=True, progress=True)
        self.deeplab.classifier[4] = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1) 

    def forward(self, x):
        return self.deeplab(x)


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, msk_dir, img_transform=None, msk_transform=None):
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.img_transform = img_transform
        self.msk_transform = msk_transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img = os.listdir(self.img_dir)[idx].split(".")[0]
        img_path = os.path.join(self.img_dir, img + ".jpg")
        image = read_image(img_path)
        image = image.to(torch.float) / 255
        
        msk_path = os.path.join(self.msk_dir, img + ".png")
        mask = read_image(msk_path)
        mask = mask.to(torch.float)[0, :, :].unsqueeze(0) / 255
        if self.img_transform:
            image = self.img_transform(image)
        if self.msk_transform:
            label = self.msk_transform(mask)
        return image, mask


def _get_train_data_loader(batch_size, training_dir, is_distributed, **kwargs):
    logger.info("Get train data loader")
    dataset = CustomImageDataset(
        img_dir=training_dir + "/train/images/",
        msk_dir=training_dir + "/train/masks/",
        img_transform=None,
        #transform=transforms.ToTensor()
        msk_transform=None
        #transform=transforms.Normalize((0.1307,), (0.3081,))
    )
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **kwargs
    )

def _get_test_data_loader(batch_size, training_dir, **kwargs):
    logger.info("Get train data loader")
    dataset = CustomImageDataset(
        img_dir=training_dir + "/test/images/",
        msk_dir=training_dir + "/test/masks/",
        img_transform=None,
        #transform=transforms.ToTensor()
        msk_transform=None
        #transform=transforms.Normalize((0.1307,), (0.3081,))
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )


def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def train(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.backend, dist.get_world_size()
            )
            + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(), args.num_gpus)
        )

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, is_distributed, **kwargs)
    test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    model = Net().to(device)
    #model = deeplabv3_resnet50(num_classes=1).to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #loss_func = nn.BCEWithLogitsLoss()#.to(device)
    loss_func = torch.nn.MSELoss(reduction='mean')
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (images, masks) in enumerate(train_loader, 1):
            print("Data: ", images.shape)
            #print("Target: ", masks.squeeze().shape)
            #print(images[0])
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(images)
            print("Output: ", output["out"].shape, " | Max: ", torch.max(output["out"]), " | Min: ", torch.min(output["out"]))
            print("Input: ", images.shape, " | Max: ", torch.max(images), " | Min: ", torch.min(images))
            print("Mask: ", masks.shape, " | Max: ", torch.max(masks), " | Min: ", torch.min(masks))
            loss = loss_func(output["out"], masks)
            print("Loss: ", loss.item())
            y_pred = output['out'].data.cpu().numpy().ravel()
            y_true = masks.data.cpu().numpy().ravel()
            loss.backward()
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()
            
            for name, metric in metrics.items():
                        if name == 'f1_score':
                            # Use a classification threshold of 0.1
                            print("F1", metric(y_true > 0, y_pred > 0.1))
                        else:
                            print("Metric", metric(y_true.astype('uint8'), y_pred))
                            
            if batch_idx % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(images),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test(model, test_loader, device)
    save_model(model, args.model_dir)


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = torch.nn.MSELoss(reduction='mean')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output["out"], target).item()  # sum up batch loss
            pred = output["out"].max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net())
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())
