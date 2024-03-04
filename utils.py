import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import random
import torchvision.models as models
from models import geocluser, netvlad
from os.path import join, isfile
from math import sqrt
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F

class Pre_process():
    """ Loading datasets, and preparing map descriptors """

    def __init__(self, opt, model=None, device=None):
        self.opt = opt
        self.model = model
        self.device = device
        self.dataset_folder = opt.dataset_dir
        self.map_feature_list = []

    def init_dataset(self):
        database_folder = join(self.dataset_folder, self.opt.place + '_ref_scale_' + self.opt.ratio)
        database_paths = sorted(glob(join(database_folder, "*.jpg"), recursive=True))
        for path in tqdm(database_paths):
            tile = cv2.imread(path)
            feature = observation_model(self.opt, tile, self.model, self.device)
            a_feature = feature.detach().cpu().numpy().tolist()
            self.map_feature_list.append(a_feature)
        map_feature = np.array(self.map_feature_list)[:, 0, :]
        if self.opt.ratio == '1.00':
            map_feature = map_feature.reshape(2800, 32768)
            map_feature = map_feature.reshape(70, 40, -1)
            # print(map_feature.shape)
        elif self.opt.ratio == '0.85':
            map_feature = map_feature.reshape(6466, 32768)
            map_feature = map_feature.reshape(106, 61, -1)
        elif self.opt.ratio == '1.35':
            map_feature = map_feature.reshape(4128, 32768)
            map_feature = map_feature.reshape(86, 48, -1)
        np.save('./experiment/features/'+self.opt.place +'/'+self.opt.place+'_feature_'+self.opt.model+'_'+self.opt.ratio +'.npy',map_feature)
        return True

class Aircraft():
    """ Defining the aircraft """

    def __init__(self):
        self.lat = 0
        self.lon = 0

    def move_to(self, lat, lon):
        self.lat = lat
        self.lon = lon

def set_model(opt, device):
    """ This is the loaded encoder model and the weights """

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    encoder_dim = 512
    encoder = models.vgg16(pretrained=True)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]

    if True:
        # if using pretrained then only train conv5_1, conv5_2, and conv5_3
        for l in layers[:-5]:
            for p in l.parameters():
                p.requires_grad = False
    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module('encoder', encoder)
    if opt.model == 'GeoCluser':
        random_tensor = torch.randn(1, 3, opt.height, opt.weight)
        temp = model.encoder(random_tensor).to(device)
        net_vlad = geocluser.NetVLAD(opt=opt, num_clusters=64, dim=encoder_dim, vladv2=False, temp=temp)
    elif opt.model == 'NetVlad':
        net_vlad = netvlad.NetVLAD(num_clusters=64, dim=encoder_dim, vladv2=False)
    model.add_module('pool', net_vlad)
    resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_ckpt, checkpoint['epoch']))
    model.eval()
    return model

def get_map_descriptors(opt):
    """ This is to get the map database (map tile descriptors) """

    dataFilename = os.path.join('./experiment/features/'+opt.place +'/'+opt.place+'_feature_'+opt.model+'_'+opt.ratio +'.npy')
    descriptors = np.load(dataFilename, allow_pickle=True)
    return descriptors

def routes_data(opt):
    """ Get the real route on the ground """

    dataFilename = os.path.join('./experiment/routes/' + opt.place + '_route.npy')
    routes = np.load(dataFilename, allow_pickle=True)
    return routes

def input_transform(opt,preprocess):
    """ Image pre-processing method """

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Resize((opt.height, opt.weight))])

def prepare_tile(opt,tile, preprocess='none'):
    """ Image preparation """

    tile = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
    tile = input_transform(opt, preprocess)(tile)
    return tile

def observation_model(opt,observation, model, device):
    """ Get input image descriptor """

    aerial = prepare_tile(opt, observation)
    with torch.no_grad():
        input = aerial.to(device)
        image_encoding = model.encoder(input)
        if opt.model=='GeoCluser' or opt.model=='NetVlad':
            aerial_descriptor = model.pool(image_encoding)
        return aerial_descriptor

def update_vo_estimate(vo_estimate, dx, dy):
    """ Update VO estimate location """

    vo_estimate[0] += dy * 0.185 / 100000   # Camera picture Y direction a pixel value of 0.185 m
    vo_estimate[1] += dx * 0.145 / 100000   # Camera picture X direction a pixel value of 0.145 m
    return vo_estimate

def lat2xy(area_range, lat, lon):
    """ Latitude and longitude to XY pixel coordinates """

    x = ((lon - area_range[2]) * 100000) / 0.419219
    y = ((lat - area_range[0]) * 100000) / 0.535
    return y, x

def xy2lal(area_range, y, x):
    """ XY pixel coordinates to latitude and longitude """

    lon = x * 0.419219 / 100000 + area_range[2]
    lat = y * 0.535 / 100000 + area_range[0]
    return lat, lon

def Calculation_error(step, estimate, error, routes, area_range):
    """ Calculation error """

    distance = sqrt(pow(estimate[0]-routes[step][1], 2)+pow(estimate[1]-((area_range[3] - routes[step][0]) + area_range[2]), 2))*100000
    error.append(distance)
    return distance, error

def Get_Average(list):
    """ Calculate the mean value """

    sum = 0
    for item in list:
        sum += item
    return sum/len(list)