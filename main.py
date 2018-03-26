import torch
from model import Hide, Reveal
from utils import DatasetFromFolder
import torch.nn.init as init
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

dataset = DatasetFromFolder('./data', crop_size=256)
dataloader = DataLoader(dataset, 32, shuffle=True, num_workers=4)

hide_net = Hide()
hide_net.apply(init_weights)

reveal_net = Reveal()
reveal_net.apply(init_weights)

criterion = nn.MSELoss()


hide_net.cuda()
reveal_net.cuda()
criterion.cuda()

optim_h = optim.Adam(hide_net.parameters(), lr=1e-3)
optim_r = optim.Adam(reveal_net.parameters(), lr=1e-3)

schedulee_h = MultiStepLR(optim_h, milestones=[100, 1000])
schedulee_r = MultiStepLR(optim_h, milestones=[100, 1000])

for epoch in range(2000):
    schedulee_h.step()
    schedulee_r.step()

    epoch_loss_h = 0.
    epoch_loss_r = 0.
    for i, (secret, cover) in enumerate(dataloader):
        secret = Variable(secret).cuda()
        cover = Variable(cover).cuda()

        optim_h.zero_grad()
        optim_r.zero_grad()

        output = hide_net(secret, cover)
        loss_h = criterion(output, cover)

        epoch_loss_h += loss_h.data[0]

        reveal_secret = reveal_net(output)
        loss_r = criterion(reveal_secret, secret)

        epoch_loss_r += loss_r.data[0]

        loss = loss_h + 0.75 * loss_r
        loss.backward()
        optim_h.step()
        optim_r.step()

        if i == 3 and epoch % 20 == 0:
            save_image(torch.cat([secret.cpu().data[:4], reveal_secret.cpu().data[:4], cover.cpu().data[:4], output.cpu().data[:4]], dim=0), filename='./result/res_epoch_{}.png'.format(epoch), nrow=4)

    print('epoch {0} hide loss: {1}'.format(epoch, epoch_loss_h))
    print('epoch {0} reveal loss: {1}'.format(epoch, epoch_loss_h))
    print('=======>>>'*5)

    if epoch > 1000 and epoch % 100 == 0:
        torch.save(hide_net.state_dict(), './checkpoint/epoch_{}_hide.pkl'.format(epoch))
        torch.save(reveal_net.state_dict(), './checkpoint/epoch_{}_reveal.pkl'.format(epoch))






























