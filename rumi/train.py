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
import numpy as np

import models.dcgan as dcgan
from utils.data import find_matching_file, get_positive, get_negative, make_arrays_equal_length
from torch.utils.data import DataLoader, Dataset, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, default='mario')
parser.add_argument('--instance', type=str)
parser.add_argument('--s', type=int)
parser.add_argument('--f', type=int)

parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
# parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
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
parser.add_argument('--problem', type=int, default=0, help='Level examples')
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
    
X_pos, _= get_positive(opt.game)
X_neg, _ = get_negative(opt.game)
# balance the dataset (remove from bigger class)
print('balance the dataset (remove from bigger class)')
X_pos, X_neg = make_arrays_equal_length(X_pos, X_neg)
print("number of playble levels after balance: ", len(X_pos))
print("number of unplayble levels after balance: ", len(X_neg))

z_dims = X_pos.shape[3] #Numer different title types
num_batches = X_pos.shape[0] / opt.batchSize

X_onehot_pos = np.rollaxis(X_pos, 3, 1)
X_onehot_neg = np.rollaxis(X_neg, 3, 1)

X_train_pos = np.zeros ( (X_pos.shape[0], z_dims, map_size, map_size) )*2
X_train_neg = np.zeros ( (X_neg.shape[0], z_dims, map_size, map_size) )*2

X_train_pos[:, 2, :, :] = 1.0  #Fill with empty space
X_train_neg[:, 2, :, :] = 1.0  #Fill with empty space

#Pad part of level so its a square
X_train_pos[:X_pos.shape[0], :, :X_pos.shape[1], :X_pos.shape[2]] = X_onehot_pos
X_train_neg[:X_neg.shape[0], :, :X_neg.shape[1], :X_neg.shape[2]] = X_onehot_neg

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

netG = dcgan.DCGAN_G(map_size, nz, z_dims, ngf, ngpu, n_extra_layers)
netG.apply(weights_init)

netD = dcgan.DCGAN_D(map_size, nz, z_dims, ndf, ngpu, n_extra_layers)
netD.apply(weights_init)


if opt.s > 0:
    epochs = opt.f - opt.s
    previous_d = f'./{opt.experiment}/{opt.game}/{opt.s}/{opt.instance}/*RD_{opt.condition}.pth'
    previous_g = f'./{opt.experiment}/{opt.game}/{opt.s}/{opt.instance}/*RG_{opt.condition}.pth'
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

main_dir = f"./{opt.experiment}/{opt.game}/{opt.f}/{opt.instance}"
if not os.path.exists(main_dir):
    try:
        os.makedirs(main_dir)
        print(f"Directory '{main_dir}' created successfully.")
    except OSError as e:
        print(f"Error: Failed to create directory '{main_dir}'. {e}")
else:
        print(f"Directory '{main_dir}' already exists.")

if not os.path.exists(f"{main_dir}/all"):
    try:
        os.makedirs(f"{main_dir}/all")
        print(f"Directory '{main_dir}/all' created successfully.")
    except OSError as e:
        print(f"Error: Failed to create directory '{main_dir}/all'. {e}")
else:
        print(f"Directory '{main_dir}/all' already exists.")

input_pos = torch.FloatTensor(opt.batchSize, z_dims, map_size, map_size)
input_neg = torch.FloatTensor(opt.batchSize, z_dims, map_size, map_size)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda and torch.cuda.is_available():
    netD.cuda()
    netG.cuda()
    input_pos = input_pos.cuda()
    input_neg = input_neg.cuda()
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


X_train_pos = torch.FloatTensor(X_train_pos)
X_train_neg = torch.FloatTensor(X_train_neg)
X_train_pos_dataset = TensorDataset(X_train_pos)
X_train_neg_dataset = TensorDataset(X_train_neg)

dataloader_pos = DataLoader(X_train_pos_dataset, batch_size=opt.batchSize, shuffle=True, drop_last=True)
dataloader_neg = DataLoader(X_train_neg_dataset, batch_size=opt.batchSize, shuffle=True, drop_last=True)
num_batches = len(dataloader_pos)
gen_iterations = 0
for epoch in range(epochs + 1):
    
    # #! data_iter = iter(dataloader)

    # length = len(X_train_pos)
    # perm = torch.randperm(length)

    # X_train_pos = X_train_pos[perm]
    # X_train_neg = X_train_neg[perm]


    i = 0
    while i < num_batches:#len(dataloader):
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
        while j < Diters and i < num_batches:#len(dataloader):
            j += 1

            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            # if (i+1)*opt.batchSize <= len(X_train_pos):
            #     data_pos = X_train_pos[i*opt.batchSize:(i+1)*opt.batchSize]
            #     data_neg = X_train_neg[i*opt.batchSize:(i+1)*opt.batchSize]
            # else:
            #     data_pos = X_train_pos[i*opt.batchSize:]
            #     data_neg = X_train_neg[i*opt.batchSize:]

            data_pos = next(iter(dataloader_pos))
            data_neg = next(iter(dataloader_neg))
            i += 1

            # if len(data_pos) == opt.batchSize and len(data_neg) == opt.batchSize:
            real_cpu_pos = torch.FloatTensor(data_pos[0])
            real_cpu_neg = torch.FloatTensor(data_neg[0])

            netD.zero_grad()
            if opt.cuda and torch.cuda.is_available():
                real_cpu_pos = real_cpu_pos.cuda()
                real_cpu_neg = real_cpu_neg.cuda()

            input_pos.resize_as_(real_cpu_pos).copy_(real_cpu_pos)
            input_neg.resize_as_(real_cpu_neg).copy_(real_cpu_neg)

            errD_real_pos = netD(Variable(input_pos))
            errD_real_neg = netD(Variable(input_neg))
            
            errD_real_pos.backward(one)
            errD_real_neg.backward(mone)

            # train with fake
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise, volatile = True) # totally freeze netG
            fake = Variable(netG(noisev).data)
            errD_fake = netD(fake)

            errD_fake.backward(mone)

            real_pos_loss = errD_real_pos
            fake_loss = 1 - errD_fake
            real_neg_loss = 1 - errD_real_neg

            # Sum all losses
            # errD = (real_pos_loss - fake_loss) + 0.5 * (real_neg_loss - fake_loss)
            errD = real_pos_loss - fake_loss - real_neg_loss
            # errD = errD_real_pos - errD_fake
            optimizerD.step()
            # else:
                # print("len(data_pos) != opt.batchSize or len(data_neg) != opt.batchSize")

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        errG = netD(fake)
        errG.backward(one)
        optimizerG.step()
        gen_iterations += 1

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real_pos: %f Loss_D_real_neg: %f Loss_D_fake %f'
            % (epoch, epochs, i, num_batches, gen_iterations,
            errD.data[0], errG.data[0], errD_real_pos.data[0], errD_real_neg.data[0], errD_fake.data[0]))

    # do checkpointing
    if epoch % 50 == 0:
        print(f"<><> saved model on epoch {epoch}")
        torch.save(netG.state_dict(), f'{main_dir}/RG_checkpoint.pth')
        torch.save(netD.state_dict(), f'{main_dir}/RD_checkpoint.pth')
torch.save(netG.state_dict(), f'{main_dir}/RG.pth')
torch.save(netD.state_dict(), f'{main_dir}/RD.pth')