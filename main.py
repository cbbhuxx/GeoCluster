import argparse
import torch
import numpy as np
import cv2
from motion_estimate import Homography as Mestimator
import particle
import warnings
import utils

num_devices = torch.cuda.device_count()
parser = argparse.ArgumentParser(description='GeoCluser-PF')
parser.add_argument('--mode', type=str, default='PF', choices=['Pre', 'PF'],help='Select mode, pre-processing or positioning')
parser.add_argument('--model', type=str, default='GeoCluser', choices=['GeoCluser', 'NetVlad'],help='The model used by the coding module')
parser.add_argument('--np', type=int, default=5000, help='Number of particles')
parser.add_argument('--place', type=str, default='Beijing', help='Test place for positioning')
parser.add_argument('--dataset_dir', type=str, default='./experiment/tiles', help='_')
parser.add_argument('--resume', type=str, default='./weights/vgg16_netvlad_checkpoint/', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--weight', type=int, default=480, help='Weight of the camera image reset')
parser.add_argument('--height', type=int, default=480, help='Height of the camera image reset')
parser.add_argument('--gpu_ids', type=str, default=str(num_devices-1), help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--steps', type=int, default=327, help='Number of steps')
parser.add_argument('--ratio', type=str, default='1.00',help='The ratio of the reference image to the input image', choices=['1.00', '0.85', '1.35'])
parser.add_argument('--visualize', action='store_true', help='if specified, print more debugging information')
area_range = [116.101049066764, 116.116820683836, 40.0652127940419, 40.0856287594884]


def localize(opt, routes, map_features):
    """ Perform PF algorithm """
    print("Particle filter experiment in {}".format(opt.place))
    pf_error, vo_error, best_error = [], [], []

    steps = opt.steps
    vo = np.zeros((steps, 2))
    robot = utils.Aircraft()

    lon, lat = routes[0]
    lon = (area_range[3] - lon) + area_range[2]
    robot.move_to(lat, lon)
    vo_estimate = np.array([lat, lon])
    aerial = cv2.imread('./experiment/query_images/' + opt.place+'_query/0.jpg')
    motion_estimator = Mestimator(aerial)
    particles = particle.init_particles(opt, area_range)

    aerial_features = utils.observation_model(opt,aerial, model, device)
    particles = particle.update_weights(opt, particles, area_range, aerial_features, map_features)
    estimate = particle.get_estimate(opt, particles)
    best_particle_index = np.argmax(particles[:, -1])
    best_particle = particles[best_particle_index, :2]
    vo[0, 0:2] = vo_estimate

    vo_distance, vo_error = utils.Calculation_error(0, vo_estimate, vo_error, routes, area_range)
    best_distance, best_error = utils.Calculation_error(0, best_particle,  best_error, routes, area_range)
    pf_distance, pf_error = utils.Calculation_error(0, estimate, pf_error, routes, area_range)

    # -------------------------------------------------------------
    for step in range(1, steps):
        lon, lat = routes[step]
        lon = (area_range[3] - lon) + area_range[2]
        robot.move_to(lat, lon)
        import time
        start = time.time()

        aerial = cv2.imread('./experiment/query_images/' + opt.place+'_query/' + str(step) + '.jpg')

        _, translation, _ = motion_estimator.estimate(aerial)
        dx, dy = -translation[0], translation[1]

        vo_estimate = utils.update_vo_estimate(vo_estimate, dx, dy)
        particles = particle.motion_update(particles, area_range, dx, dy)

        aerial_features = utils.observation_model(opt, aerial, model, device)
        particles = particle.update_weights(opt, particles, area_range, aerial_features, map_features)
        neff = 1.0 / np.sum(np.power(particles[:, -1], 2))        # Number of effective particles
        if neff < 2 * particles.shape[0] / 3:  # resample
            particles = particle.systematic_resample(particles)   # Get indices of particles to resample

        estimate = particle.get_estimate(opt, particles)
        best_particle_index = np.argmax(particles[:, -1])
        best_particle = particles[best_particle_index, :2]

        time0 = time.time() - start

        if not opt.visualize:
            particle.visualize(aerial, opt, step, robot, particles, estimate, vo_estimate, best_particle)

        vo_distance, vo_error = utils.Calculation_error(step, vo_estimate, vo_error, routes, area_range)
        pf_distance, pf_error = utils.Calculation_error(step, estimate, pf_error, routes, area_range)
        best_distance, best_error = utils.Calculation_error(step, best_particle, best_error, routes, area_range)

        # ---------------------------------------------
        print('\033[31mExecute to step {}\033[0m'.format(step))
        print('\033[33mNumber of particles left {}\nSingle step execution time {} sec\033[0m'.format(particles.shape[0], time0))

        print(
            "Estimated Location {}\nReal Location [{} {}]\nError is {} meters\n"
            .format(estimate[0:2], routes[step][1],
                    (area_range[3] - routes[step][0]) + area_range[2], pf_distance))
    print("pf_mean_error: {} meters\nvo_mean_error: {} meters\nbest_mean_error: {} meters\n"
          .format(utils.Get_Average(pf_error), utils.Get_Average(vo_error), utils.Get_Average(best_error)))
    print("Total {} steps".format(steps))


if __name__ == "__main__":

    warnings.filterwarnings('ignore')       # Ignore warnings

    opt = parser.parse_args()
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
    particles = np.zeros((opt.np, 3))       # Latitude, longitude, weights
    model = utils.set_model(opt, device)               # Add code model
    if opt.mode == 'Pre':
        res = utils.Pre_process(opt, model, device).init_dataset()
    map_features = utils.get_map_descriptors(opt)    # Get the map database
    routes = utils.routes_data(opt)
    localize(opt, routes, map_features)
