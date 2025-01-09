import os
import glob
import torch
import pydicom
import argparse

import numpy as np
import torch.nn as nn
import torchvision.transforms

import torchvision
import torchvision.transforms as transforms

from network import Model
from utilities import normalize_image, read_image, max, min 


parser = argparse.ArgumentParser(description='Input clinical details')
parser.add_argument('--gender', type=list, default=[0], help="male = 0, female = 1")
parser.add_argument('--age', type=list, default=[64.0], help="age")
parser.add_argument('--history_of_diabetes', type=list, default=[0], help="without = 0, with = 1")
parser.add_argument('--history_of_hypertension', type=list, default=[1], help="without = 0, with = 1")
parser.add_argument('--smoking_history', type=list, default=[1], help="without = 0, with = 1")
parser.add_argument('--drinking_history', type=list, default=[0, 0, 0, 1], 
                    help="without = [1, 0, 0, 0]; sometimes = [0, 1, 0, 0]; frequently = [0, 0, 1, 0]; else = [0, 0, 0, 1]")
parser.add_argument('--family_history_of_tumor', type=list, default=[0], 
                    help="without = 0, with = 1")

parser.add_argument('--pathological_stage', type=list, default=[0, 1, 0, 0, 0], 
                    help="I = [1, 0, 0, 0, 0], II = [0, 1, 0, 0, 0]; III = [0, 0, 1, 0, 0]; IV = [0, 0, 0, 1, 0]; else = [0, 0, 0, 0, 1]")
parser.add_argument('--perineural_invasion', type=list, default=[0, 0, 1], 
                    help="without = [1, 0, 0], with = [0, 1, 0], else = [0, 0, 1]")
parser.add_argument('--pathological_type', type=list, default=[1, 0, 0, 0], 
                    help="well = [1, 0, 0, 0], mix = [0, 1, 0, 0], poor = [0, 0, 1, 0], else = [0, 0, 0, 1]")
parser.add_argument('--position', type=list, default=[1, 0, 0, 0], 
                    help="RCC = [1, 0, 0, 0]; LCC = [0, 1, 0, 0], REC = [0, 0, 1, 0], else = [0, 0, 0, 1]")
parser.add_argument('--white_blood_cell_count', type=list, default=[6.1], help="*109/L")
parser.add_argument('--red_blood_cell_count', type=list, default=[4.19], help="*109/L")
parser.add_argument('--hemoglobin', type=list, default=[124.0], help="g/L")
parser.add_argument('--platelet_concentration', type=list, default=[153.0], help="*109/L")
parser.add_argument('--neutrophil_count', type=list, default=[3.8], help="*109/L")
parser.add_argument('--lymphocyte_count', type=list, default=[1.9], help="*109/L")
parser.add_argument('--monocyte_count', type=list, default=[0.3], help="*109/L")
parser.add_argument('--red_cell_volumn_distribution_width', type=list, default=[14.4])
parser.add_argument('--plateletcrit', type=list, default=[0.2])
parser.add_argument('--mean_platelet_volume', type=list, default=[11.3])
parser.add_argument('--albumin', type=list, default=[42.8])
parser.add_argument('--globulin', type=list, default=[20.8])
parser.add_argument('--albumin_globulin_ratio', type=list, default=[2.1])
parser.add_argument('--blood_glucose', type=list, default=[5.03])
parser.add_argument('--triglyceride', type=list, default=[1.23])
parser.add_argument('--cholesterol', type=list, default=[4.42])
parser.add_argument('--high_density_lipoprotein', type=list, default=[1.61])
parser.add_argument('--low_density_lipoprotein', type=list, default=[2.46])
parser.add_argument('--carcinoembryonic_antigen', type=list, default=[0.2])
parser.add_argument('--carcinoembryonic_antigen_199', type=list, default=[2.29])
parser.add_argument('--carcinoembryonic_antigen_125', type=list, default=[1.08])
args = parser.parse_args()


def main():

    test_aug2 = torchvision.transforms.Compose([
        transforms.Resize([384, 384]),
        transforms.Normalize((0.485, 0.485, 0.485), (0.224, 0.224, 0.224)),
    ])

    # for model construction
    model = Model()

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('./weights/parameters.pth', map_location=torch.device('cpu')))
    model = model.eval().cuda()

    files = glob.glob(os.path.join('./data/M0001', '*', '*.dcm'))
    files = sorted(files)

    CT_images = []
    for file in files[2:]:
        img = read_image(file, None)[None, :]
        CT_images.append(img)
    CT_images = torch.cat(CT_images, dim=1)
    CT_images = test_aug2(CT_images)


    # for shaolun designing the web server
    clinical_data = np.array(args.gender + args.age + args.history_of_diabetes + args.history_of_hypertension + args.smoking_history + args.drinking_history + args.family_history_of_tumor + args.pathological_stage + args.perineural_invasion + args.pathological_type + args.position + args.white_blood_cell_count
                               + args.red_blood_cell_count + args.hemoglobin + args.platelet_concentration + args.neutrophil_count + args.lymphocyte_count + args.monocyte_count + args.red_cell_volumn_distribution_width + args.plateletcrit + args.mean_platelet_volume + args.albumin
                               + args.globulin + args.albumin_globulin_ratio + args.blood_glucose + args.triglyceride + args.cholesterol + args.high_density_lipoprotein + args.low_density_lipoprotein + args.carcinoembryonic_antigen + args.carcinoembryonic_antigen_199 + args.carcinoembryonic_antigen_125)



    clinical_data = (clinical_data - min) / (max - min)

    data = (CT_images.cuda(), torch.Tensor(clinical_data[None, ]).cuda())

    prediction = model(data)
    # print(prediction)
    prediction = torch.sigmoid(prediction)
    print('The final MSI-H probability {}'.format(prediction[0]))


if __name__ == '__main__':
    main()
