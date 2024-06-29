import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import pdb
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import numpy as np

from torch.utils.data import DataLoader, Dataset, TensorDataset


import models.cdcgan as cdcgan
from utils.file import make_sure_dir_exists
from utils.data import find_matching_file, get_positive, get_negative, get_positive_db, get_negative_db


parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, default='mario')
parser.add_argument('--instance', type=str)
parser.add_argument('--cond', type=int, default=1)
parser.add_argument('--s', type=int, default=0)
parser.add_argument('--f', type=int)

parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')

parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default='./trained_models', help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
opt = parser.parse_args()
print(opt)
        

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

print("CUDA available: ", torch.cuda.is_available())
print("CUDA device count: ", torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
 
map_size = 64
if opt.game == "mario" or opt.game == "cave_treasures":
    if opt.cond == 1:
        X_pos = get_positive_db(opt.game, 1)
        y_pos = np.full(len(X_pos), 1)

        X_neg_0 = get_positive_db(opt.game, 2)
        X_neg_4 = get_positive_db(opt.game, 3)

    elif opt.cond == 2:
        X_pos = get_positive_db(opt.game, 2)
        y_pos = np.full(len(X_pos), 2)


        X_neg_0 = get_positive_db(opt.game, 1)
        X_neg_4 = get_positive_db(opt.game, 3)
    elif opt.cond == 3:
        X_pos = get_positive_db(opt.game, 3)
        y_pos = np.full(len(X_pos), 3)

        X_neg_0 = get_positive_db(opt.game, 1)
        X_neg_4 = get_positive_db(opt.game, 2)

    y_neg_0 = np.full(len(X_neg_0), 1)
    y_neg_4 = np.full(len(X_neg_4), 1)

    X_neg = np.concatenate((X_neg_0, X_neg_4))
    y_neg = np.concatenate((y_neg_0, y_neg_4))

X = np.concatenate((X_pos, X_neg))
y = np.concatenate((y_pos, y_neg))

z_dims = X_pos.shape[3] #Numer different title types

shape = y.shape
if len(shape) == 1:
    print("Shape is (x,)")
    y_dims = 1
elif len(shape) == 2 and shape[1] == 2:
    print("Shape is (x,2)")
    y_dims = 2
else:
    print("Shape is not (x,) or (x,2)")

print ("SHAPE ",X.shape) 

X_onehot = np.rollaxis(X, 3, 1)
print ("SHAPE ",X_onehot.shape)    #(173, 14, 28, 16)

X_train = np.zeros ( (X.shape[0], z_dims, map_size, map_size) )*2

X_train[:, 2, :, :] = 1.0  #Fill with empty space

#Pad part of level so its a square
X_train[:X.shape[0], :, :X.shape[1], :X.shape[2]] = X_onehot
y_train = y

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG = cdcgan.DCGAN_G(map_size, nz + y_dims, z_dims, ngf, ngpu, n_extra_layers)
netG.apply(weights_init)

netD = cdcgan.DCGAN_D(map_size, nz, z_dims + y_dims, ndf, ngpu, n_extra_layers)
netD.apply(weights_init)

if opt.s > 0:
    epochs = opt.f - opt.s
    previous_g = f"./{opt.experiment}/{opt.game}/{opt.instance}/{opt.s}/CG*.pth"
    previous_d = f"./{opt.experiment}/{opt.game}/{opt.instance}/{opt.s}/CD*.pth"
    
    matching_files_d = find_matching_file(previous_d)
    if len(matching_files_d) > 0:
        matching_files_d = matching_files_d[0]
        netD.load_state_dict(torch.load(matching_files_d))
        print(netD)
    matching_files_g = find_matching_file(previous_g)
    if len(matching_files_g) > 0:
        matching_files_g = matching_files_g[0]
        netG.load_state_dict(torch.load(matching_files_g))
        print(netG)
else:
    epochs = opt.f

main_dir = f"./{opt.experiment}/{opt.game}/{opt.instance}"
make_sure_dir_exists(f"{main_dir}")

input = torch.FloatTensor(opt.batchSize, z_dims, map_size, map_size)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda and torch.cuda.is_available():
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    print("Using ADAM")
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
train_dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, drop_last=True)
num_batches = len(dataloader)
gen_iterations = 0
for epoch in range(epochs + 1):
    for data, label  in dataloader:
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters

        j = 0
        while j < Diters:
            j += 1

            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            data, label = next(iter(dataloader))
            # i += 1

            real_cpu = data
            real_label = label

            netD.zero_grad()

            if opt.cuda and torch.cuda.is_available():
                real_cpu = real_cpu.cuda()
                real_label = real_label.cuda()

            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)
            labelv = Variable(real_label)
            errD_real = netD(inputv, labelv)
            errD_real.backward(one)

            # train with fake
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise, volatile = True) # totally freeze netG
            fake = Variable(netG(noisev, labelv).data)
            inputv = fake
            errD_fake = netD(inputv, labelv)
            errD_fake.backward(mone)
            errD = errD_real - errD_fake
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()

        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, labelv)
        errG = netD(fake, labelv)
        errG.backward(one)
        optimizerG.step()
        gen_iterations += 1

        print('[%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (opt.s + epoch, opt.s + epochs, gen_iterations,
            errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

    if epoch % 100 == 0 and epoch > 0:
        make_sure_dir_exists(f"{main_dir}/{epoch + opt.s}")

        print(f"<><> saved model on epoch {opt.s + epoch}")
        torch.save(netG.state_dict(), f'{main_dir}/{opt.s + epoch}/CG_{opt.cond}_checkpoint_{opt.s + epochs}.pth')
        torch.save(netD.state_dict(), f'{main_dir}/{opt.s + epoch}/CD_{opt.cond}_checkpoint_{opt.s + epochs}.pth')
    
make_sure_dir_exists(f"{main_dir}/{opt.f}")
torch.save(netG.state_dict(), f'{main_dir}/{opt.f}/CG_{opt.cond}.pth')
torch.save(netD.state_dict(), f'{main_dir}/{opt.f}/CD_{opt.cond}.pth')