import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pdb
import torch
import torchvision.utils as vutils
from torch.autograd import Variable
from datetime import datetime
import subprocess
import numpy
import models.cdcgan as cdcgan
from utils.data import map_output_to_symbols
import argparse
from utils.data import find_matching_file, get_reach_move, get_cols_rows, get_z_dims
import numpy as np

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=str, default=10000)
    parser.add_argument('--game', type=str, default='mario')
    parser.add_argument('--instance', type=str)
    parser.add_argument('--directory', type=str, default='./out')
    parser.add_argument('--image', type=bool, default=False)
    parser.add_argument('--solution', type=bool, default=False)
    parser.add_argument('--batchsize', type=int, default=100)

    opt = parser.parse_args()

    nz = 32
    

    modelToLoad = f"{opt.directory}/models/exp1/{opt.game}/{opt.instance}/{opt.epochs}/CG*.pth"
    matching_files = find_matching_file(modelToLoad)
    if len(matching_files)  > 0:
        print(matching_files)
        matching_files = matching_files[0]
    batch_size = opt.batchsize
    #nz = 10 #Dimensionality of latent vector

    imageSize = 64
    ngf = 64
    ngpu = 1
    n_extra_layers = 0
    z_dims = get_z_dims(opt.game) #number different titles

    y_dims = 1

    generator = cdcgan.DCGAN_G(imageSize, nz + y_dims, z_dims, ngf, ngpu, n_extra_layers)

    generator.load_state_dict(torch.load(matching_files, map_location=lambda storage, loc: storage))

    lv = torch.randn(batch_size, nz, 1, 1, device=device)
    latent_vector = torch.FloatTensor( lv ).view(batch_size, nz, 1, 1) 

    # labels = torch.zeros(batch_size, 3)
    # labels[:, cond] = 1  # Set the first column to 1
    # labels = torch.ones(batch_size)
    
    # labels = torch.FloatTensor(np.tile([[1, 0]], (batch_size, 1)))
    labels = torch.full((batch_size,), 0)

    levels = generator(Variable(latent_vector, volatile=True), Variable(labels))

    #levels.data = levels.data[:,:,:14,:28] #Cut of rest to fit the 14x28 tile dimensions

    level = levels.data.cpu().numpy()
    cols,rows = get_cols_rows(opt.game)
    level = level[:,:,:cols,:rows]
    level = numpy.argmax( level, axis = 1)

    directory = f"{opt.directory}/artifacts/exp1/{opt.game}/{opt.instance}/{opt.epochs}/C"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")
        
    for i in range(batch_size):
        output = level[i]
        result = map_output_to_symbols(opt.game, output)
        result_string = ""
        for row in result:
            for item in row:
                result_string += item
            result_string += "\n"
            
        with open(directory + "/" + str(i) + ".lvl", "w") as file:
            # Write text to the file
            file.write(result_string)

        if opt.solution == True:
            try:
                reach_move = get_reach_move(opt.game)
                script_path = './sturgeon/level2repath.py'
                arguments = ['--outfile', directory + "/" + str(i) + ".path.lvl",'--textfile', directory + "/" + str(i) + ".lvl",'--reach-connect', "--src { --dst } --move " + reach_move]
                command = ['python', script_path] + arguments
                print(command)
                result = subprocess.run(command, check=True)
                if os.path.exists(directory + "/" + str(i) + ".path.lvl"):
                    print("Path exists. Level Playble.")
                    vis_path = directory + "/" + str(i) + ".path.lvl"
                else:
                    print("Path does not exist. Level Unplayble.")
                    vis_path = directory + "/" + str(i) + ".lvl"
            except subprocess.CalledProcessError:
                print("Path does not exist. Level Unplayble.")
                vis_path = directory + "/" + str(i) + ".lvl"

        if opt.image == True:
            script_path = './level2image/level2image.py'
            arguments = [vis_path,
                        '--fmt', 'png']
            command = ['python', script_path] + arguments
            subprocess.run(command, check=True)
