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
from utils.constants import MARIO_COLS, MARIO_ROWS
import argparse
from utils.data import find_matching_file
import numpy as np

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=str, default=400)
    parser.add_argument('--game', type=str, default='mario')
    parser.add_argument('--instance', type=str)

    opt = parser.parse_args()

    nz = 32
    

    modelToLoad = f"../../../scratch/bazzaz.ma/NegativeExample/models/{opt.game}/{opt.epochs}/{opt.instance}/*CG*.pth"
    matching_files = find_matching_file(modelToLoad)
    if len(matching_files)  > 0:
        print(matching_files)
        matching_files = matching_files[0]
    batch_size = 1000
    #nz = 10 #Dimensionality of latent vector

    if opt.game == "cave":
        imageSize = 32
    elif opt.game == "mario":
        imageSize = 64
    ngf = 64
    ngpu = 1
    n_extra_layers = 0
    z_dims = 10 #number different titles
    if opt.neg is True:
        y_dims = 2
    else:
        y_dims = 1

    generator = cdcgan.DCGAN_G(imageSize, nz + y_dims, z_dims, ngf, ngpu, n_extra_layers)

    generator.load_state_dict(torch.load(matching_files, map_location=lambda storage, loc: storage))

    lv = torch.randn(batch_size, 32, 1, 1, device=device)
    latent_vector = torch.FloatTensor( lv ).view(batch_size, nz, 1, 1) 

    # labels = torch.zeros(batch_size, 3)
    # labels[:, cond] = 1  # Set the first column to 1
    # labels = torch.ones(batch_size)
    if opt.neg is True:
        labels = torch.FloatTensor(np.tile([[1, opt.condition]], (batch_size, 1)))
    else:
        labels = torch.full((batch_size,), opt.condition)
    levels = generator(Variable(latent_vector, volatile=True), Variable(labels))

    #levels.data = levels.data[:,:,:14,:28] #Cut of rest to fit the 14x28 tile dimensions

    level = levels.data.cpu().numpy()
    level = level[:,:,:MARIO_COLS,:MARIO_ROWS] #Cut of rest to fit the 14x28 tile dimensions
    level = numpy.argmax( level, axis = 1)

    directory = f"../../../scratch/bazzaz.ma/NegativeExample/artifacts/{opt.game}/{opt.epochs}/{opt.instance}/C"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")
        
    for i in range(batch_size):
        output = level[i]
        result = map_output_to_symbols("mario", output)
        result_string = ""
        for row in result:
            for item in row:
                result_string += item
            result_string += "\n"
            
        with open(directory + "/" + str(i) + ".lvl", "w") as file:
            # Write text to the file
            file.write(result_string)

        try:
            reach_move = "platform"
            script_path = './sturgeon/level2repath.py'
            arguments = ['--outfile', directory + "/" + str(i) + ".path.lvl",'--textfile', directory + "/" + str(i) + ".lvl",'--reach-move', reach_move]
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

        
        script_path = './level2image/level2image.py'
        arguments = [vis_path,
                    '--fmt', 'png']
        command = ['python', script_path] + arguments
        subprocess.run(command, check=True)
