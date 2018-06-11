from pretrainedmodels import resnet152, resnet18
from torch.utils.data import DataLoader
from torchvision.transforms import *
from training.training import Trainer
from generic_utils.metrics import mse
from generic_utils.output_watchers import RegressionWatcher
from pretrainedmodels.models import resnet50
from dataset.dataset import *
import torch.nn as nn

from utils import CsvAvitoProvider

trainer = Trainer('avito', lambda x, y: torch.sqrt(nn.MSELoss().cuda()(x, y)),
                  lambda x, y: pow(mse(x, y), 0.5))  # fixme

trainer.set_output_watcher(RegressionWatcher(trainer.watcher))
# proper losses

DATA = '/mnt/data/competition_files/train_jpg/'

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
BATCH_SIZE = 256
tsize = 224
EPOCH_NUM = 200

train_transform = Compose([  # CenterCrop(100),
    RandomResizedCrop(size=tsize, scale=(0.7, 1)),
    RandomRotation(degrees=20),
    ColorJitter(0.5, 0.1, 0.1, 0.1),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    Resize((tsize, tsize)),
    ToTensor(),
    Normalize(rgb_mean, rgb_std)])

test_transform = Compose([Resize((tsize, tsize)),
                          ToTensor(), Normalize(rgb_mean, rgb_std)])

provider = CsvAvitoProvider('/mnt/img2deal.csv', DATA)
ds = FolderCachingDataset(train_transform, test_transform, provider)

print('load success')

model = resnet50(pretrained='imagenet')
model.last_linear = nn.Sequential(
    nn.Linear(model.last_linear.in_features, 1024),
    nn.ReLU(),
    nn.Linear(512, 1),
    nn.Sigmoid()
)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
lr = 1e-4

num_parameters = sum([l.nelement() for l in model.parameters()])
print('number of parameters: {}'.format(num_parameters))

# print('loading')
# model.load_state_dict(torch.load('checkpoints/mobilenetv2_30_loss_0.806.pth.tar')['state_dict'])
# print('success')


optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.975)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)


save_dir = 'checkpoints'
try:
    os.makedirs(save_dir)
except:
    pass


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    learning_rate = lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=24)
ds.setmode('val')
val_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=24, shuffle=False)
ds.setmode('train')

best_acc = 0
for epoch in range(0, EPOCH_NUM):
    ds.setmode('train')
    loss, metric_val = trainer.train(train_loader, model, optimizer, epoch)
    ds.setmode('val')
    trainer.validate(val_loader, model)
    adjust_learning_rate(optimizer, epoch)

    is_best = metric_val > best_acc
    best_acc = max(metric_val, best_acc)
    if epoch % 10 == 0 and is_best:
        trainer.save_checkpoint({
            'epoch': epoch + 1,
            'metric': metric_val,
            'loss': loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch, loss, filename=osp.join(save_dir, 'resnet_18_{}_loss_{:.3f}.pth.tar'))
