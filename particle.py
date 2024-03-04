import numpy as np
from sklearn.metrics import pairwise_distances
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
from utils import lat2xy

area_range = [116.101049066764, 116.116820683836, 40.0652127940419, 40.0856287594884]

def init_particles(opt, area_range):
    """ Initialize the state of the particle """

    lat = np.random.uniform(area_range[0], area_range[1], opt.np)
    lon = np.random.uniform(area_range[2], area_range[3], opt.np)
    weights = 1 / opt.np * np.ones(opt.np)
    states = np.stack([lat,lon,weights],axis=1)
    return states

def get_estimate(opt, particles):
    """ Calculate estimated position """

    sorted_idx = np.argsort(particles[:, 2])[::-1]
    nparticles = particles.shape[0]
    if nparticles > opt.np/8:           # When the number of examples is greater than 1/8 of the total,the 10 particles
        nparticles = 10                 # with the greatest weight are selected as the target of localization estimation

    sorted_idx = sorted_idx[0:nparticles]
    particles = particles[sorted_idx, :]

    mean_lat = np.average(particles[:, 0], weights=particles[:, -1], axis=0)
    mean_lon = np.average(particles[:, 1], weights=particles[:, -1], axis=0)
    weighted_mean = np.asarray([mean_lat, mean_lon])
    return weighted_mean

def motion_update(particles, area_range, dx, dy):
    """ This method updates particles states """

    dlon_m = dx * 0.145 * 0.00001
    dlat_m = dy * 0.185 * 0.00001

    particles[:, 0] += dlat_m
    particles[:, 1] += dlon_m

    # Kill particles outside bbox
    a = np.greater(particles[:, 0], area_range[1])
    c = np.greater(particles[:, 1], area_range[3])
    b = np.less(particles[:, 0], area_range[0])
    d = np.less(particles[:, 1], area_range[2])
    mask = (1 - a) * (1 - b) * (1 - c) * (1 - d)
    particles[:, 2] *= mask
    return particles

def update_weights(opt, particles, area_range, aerial_features, map_features):
    """ Update particle's weigths"""

    W, H, _ = map_features.shape
    y, x = lat2xy(area_range, particles[:, 0], particles[:, 1])

    if opt.ratio == "1.00":             # The area where the center point of the map tile is located
        y = (y - 250) * H / 2410 - 1    # y = (y-tile.height) * H / map.height - 1
        x = (x - 370) * W / 4130 - 1    # x = (x-tile.weight) * W / map.weight - 1
    elif opt.ratio == "0.85":
        y = (y - 235) * H / 2440 - 1
        x = (x - 315) * W / 4240 - 1
    elif opt.ratio == "1.35":
        y = (y - 375) * H / 2160 - 1
        x = (x - 500) * W / 3870 - 1
    else:
        raise Exception('Unknown ratio')

    x = np.around(x).astype(int)
    y = np.around(y).astype(int)

    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)

    aerial_features = aerial_features.cpu().numpy()

    descriptors = map_features[x, H-1 - y]

    distances = pairwise_distances(descriptors, aerial_features).squeeze()
    probs = (2 - distances) / 2

    particles[:, -1] *= probs
    sp = particles[:, -1].sum()
    particles[:, -1] /= sp

    return particles

def systematic_resample(particles):
    """ Get indices of particles to resample using systematic method """

    # Remove 50 % of particles if needed
    sorted_idx = np.argsort(particles[:, -1])[::-1]
    nparticles = particles.shape[0]

    if nparticles > 500:
        nparticles = int(0.5 * particles.shape[0])
        nparticles = max(nparticles, 500)

        sorted_idx = sorted_idx[0:nparticles]
        particles = particles[sorted_idx, :]
        sp = particles[:, -1].sum()
        particles[:, -1] /= sp

    # resample
    sample = np.random.rand(1)
    arange = np.arange(0, particles.shape[0])
    positions = (sample + arange) / particles.shape[0]

    indexes = np.zeros((particles.shape[0]))
    cumulative_sum = np.cumsum(particles[:, -1], axis=0)
    cumulative_sum[-1] = 1.0
    i, j = 0, 0
    while i < particles.shape[0]:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    particles = particles[indexes.astype(np.int64), :]
    particles[:, -1] = 1.0 / particles.shape[0]

    return particles


def visualize(aerial,opt, step, robot, particles, estimate, vo_estimate, best_particle=None):
    """ Visualization vehicles and particles in the world """
    plt.figure(figsize=(6, 8), dpi=100)
    area_map = cv2.imread('./experiment/map/' + opt.place + '.tif')
    plt.subplot(2, 1, 1)
    plt.title('Particle filter test in Beijing, step {}'.format(step), fontsize=14)
    grid = [0, area_map.shape[1], 0, area_map.shape[0]]
    
    plt.imshow(cv2.cvtColor(area_map, cv2.COLOR_BGR2RGB), extent=grid)
    plt.autoscale(False)
    plt.xticks([])
    plt.yticks([])

    # Draw particles
    if particles is not None:
        particles = particles[particles[:, -1].argsort(), :]
        y, x = lat2xy(area_range, particles[:, 0], particles[:, 1])
        plt.scatter(x, y, s=3, c=particles[:, -1], cmap='Blues',
                    norm=colors.Normalize(0, 1 / particles.shape[0]))

    # draw drone position
    y_robot, x_robot = lat2xy(area_range, robot.lat, robot.lon)
    robot_marker = plt.scatter(x_robot, y_robot, color='red', alpha=1.0)

    # Draw vo estimated pos
    # y_vo, x_vo = lat2xy(area_range, vo_estimate[0], vo_estimate[1])
    # vo_marker = plt.scatter(x_vo, y_vo, color='black', alpha=1.0)

    # Draw estimate position
    y_estimate, x_estimate = lat2xy(area_range, estimate[0], estimate[1])
    estimate_marker = plt.scatter(x_estimate, y_estimate, color='yellow', alpha=1.0)

    # legend = ['GT','PF', 'BT', 'VO']
    legend = ['Ground truth (GT)', 'Estimate(VO+NetVLAD+Ours)']
    # draw best particle
    # y_best, x_best = lat2xy(area_range, best_particle[0], best_particle[1])
    # best_marker = plt.scatter(x_best, y_best, color='purple', alpha=0.8)

    # markers = [robot_marker, estimate_marker, best_marker, vo_marker]
    markers = [robot_marker, estimate_marker]
    plt.legend(markers, legend, loc='upper right', fontsize=8)

    plt.subplot(2, 2, 3)
    # grid_q = [0, int(area_map.shape[1] / 2.5), 0, int(area_map.shape[0]/2.5)]
    plt.imshow(cv2.cvtColor(aerial, cv2.COLOR_BGR2RGB))
    plt.title("Aircraft's view", fontsize=14)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 4)
    y_estimate = area_map.shape[0]-y_estimate
    estimate_ref = area_map[int(y_estimate-235):int(y_estimate+235), int(x_estimate-315):int(x_estimate+315)]
    plt.imshow(cv2.cvtColor(estimate_ref, cv2.COLOR_BGR2RGB))
    plt.title('Estimated map location', fontsize=14)
    plt.xticks([])
    plt.yticks([])

    plt.show()
    





