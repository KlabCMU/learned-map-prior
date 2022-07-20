import numpy as np
import torch
import scipy.stats
from .utils import sample, passed_through_wall
from ..map import processing


class ParticleFilter:
    def __init__(self, config, init_state,
                 traj_net, deep_map):
        self.state = init_state
        self.num_particles = config.filter.num_particles
        self.init_cov = config.filter.init_std * np.eye(len(init_state))
        self.odom_cov = config.filter.odom_std * np.eye(len(init_state))
        self.building = config.building
        self.map = processing.get_map_by_name(config.building)
        self.traj_net = traj_net
        self.deep_map = deep_map
        self.map_prior = None
        self.allow_ble_update = config.filter.allow_ble_update
        self.allow_reinit = config.filter.allow_reinit
        self.reinit_percent = config.filter.reinit_percent
        self.reinit = False

    def init_particles(self):
        self.particles = np.zeros(
            (self.num_particles, len(self.state)))
        for id in range(self.num_particles):
            self.particles[id, :] = sample(self.init_cov) + self.state

        self.weights = np.ones(self.num_particles) * 1 / self.num_particles

    def reinit_particles(self):
        reinit_loc = processing.world_to_image_coords(
            self.state[np.newaxis, :], self.building).squeeze()
        resample_size = 50

        lower_y = reinit_loc[1] - \
            resample_size if reinit_loc[1] - resample_size > 0 else 0
        lower_x = reinit_loc[0] - \
            resample_size if reinit_loc[0] - resample_size > 0 else 0

        upper_y = reinit_loc[1] + resample_size if reinit_loc[1] + \
            resample_size < self.map_prior.shape[1] else self.map_prior.shape[1] - 1
        upper_x = reinit_loc[0] + resample_size if reinit_loc[0] + \
            resample_size < self.map_prior.shape[0] else self.map_prior.shape[0] - 1
        yv, xv = np.meshgrid(np.linspace(lower_y, upper_y, upper_y - lower_y + 1),
                             np.linspace(lower_x, upper_x, upper_x - lower_x + 1))

        coords = np.array((xv, yv))
        coords = coords.reshape(2, -1).T

        coord_weights = self.map_prior[coords[:, 0].astype(
            int), coords[:, 1].astype(int)].reshape(1, -1).T
        coord_weights[coord_weights < 0.] = 0.
        coord_weights = coord_weights/np.sum(coord_weights)
        image_points_idxs = np.random.choice(list(range(coords.shape[0])),
                                             size=self.num_particles,
                                             replace=False,
                                             p=coord_weights.squeeze())

        image_points = coords[image_points_idxs]
        self.weights = coord_weights[image_points_idxs].squeeze()
        self.weights = self.weights / np.sum(self.weights)
        self.particles = processing.image_to_world_coords(
            image_points, self.building)
        self.reinit = True

    def predict(self, odom):
        bad_point_count = 0
        for id, particle in enumerate(self.particles):
            self.particles[id, :] = particle + odom + sample(self.odom_cov)
            if self.allow_reinit:
                bad_point_count += int(passed_through_wall(particle,
                                       self.particles[id, :], self.map, self.building))
        if(bad_point_count > int(self.reinit_percent * self.num_particles)):
            self.reinit = True

    def map_sensor_model(self, particle, deep_map):
        particle = processing.world_to_image_coords(
            particle[np.newaxis, :2], self.building)[0]
        if particle[0] >= deep_map.shape[0] or particle[1] >= deep_map.shape[1] or particle[0] < 0 or particle[1] < 0:
            prob = 0
        else:
            # avoid non-positive weights
            prob = max(deep_map[particle[0], particle[1]], 0)
        return prob

    def update(self, odom_history, ble_pred):
        odom_history_im = processing.world_to_image_coords(
            odom_history[:, :2], self.building)
        odom_history_im = odom_history_im - odom_history_im[0]
        odom_history_im = torch.from_numpy(odom_history_im[np.newaxis, ...])
        self.map_prior = self.traj_net(
            odom_history_im.float(), self.deep_map.float()).detach().numpy()

        for id, particle in enumerate(self.particles):
            self.weights[id] = self.map_sensor_model(particle, self.map_prior)
            if self.allow_ble_update:
                self.weights[id] += scipy.stats.norm.pdf(
                    np.linalg.norm(particle - ble_pred), 0, 3)
                self.weights[id] = self.weights[id]/2
        self.weights[:] = self.weights[:] / np.sum(self.weights)

    def resample(self):
        particle_ids = np.arange(self.num_particles)
        selected = np.random.choice(
            particle_ids, self.num_particles, replace=True, p=self.weights)
        new_particles = np.copy(self.particles)
        for id, sid in enumerate(selected):
            new_particles[id] = self.particles[sid]
            self.weights[id] = 1 / self.num_particles
        self.particles = new_particles

    def compute_state(self):
        self.state = np.array([0., 0.])
        for id, particle in enumerate(self.particles):
            self.state[0] = self.state[0] + self.weights[id] * particle[0]
            self.state[1] = self.state[1] + self.weights[id] * particle[1]
