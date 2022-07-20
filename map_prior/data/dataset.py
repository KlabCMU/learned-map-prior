import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from . import MapConvolve

from scipy import ndimage

class MapPaddedDataset(Dataset):
    def __init__(self, dataset, omap, 
                 map_segment_size = (400,400), 
                 transform = False, 
                 infeasible_space_weight = None,
                 rotate = False):

        # trajectory data - windows, but in map coords (not normalized to zero)
        self.data = dataset
        self.transform = transform
        self.rotate = rotate
        self.omap = omap
        self.omap[self.omap == 0] = -1
        self.map_segment_size = map_segment_size
        self.max_map_padding_amount = 50 #pixels
        self.mc = MapConvolve(omap)
        self.infeasible_space_weight = infeasible_space_weight

    def __len__(self):
        return len(self.data["GT"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        kernel = None
        GT = None

        traj = self.data["pred"][idx]
        GT_traj = self.data["GT"][idx]
        
        normalized_traj = traj - traj[0]

        segment_edges = self.select_window_for_traj(traj)
        


        base_segment_size = segment_edges[1] - segment_edges[0]
        
        max_extra_map = self.map_segment_size - base_segment_size
        extra_map_below = np.random.randint(max_extra_map)
        extra_map_below[extra_map_below > self.max_map_padding_amount] = self.max_map_padding_amount
        extra_map_above = np.random.randint(max_extra_map - extra_map_below)
        extra_map_above[extra_map_above > self.max_map_padding_amount] = self.max_map_padding_amount
        segment_edges[0] -= extra_map_below
        segment_edges[1] += extra_map_above
        
        
        map_segment = self.gen_map_segment(segment_edges).astype(np.float32)

        
        GT = self.gen_ground_truth_image(traj, segment_edges, idx).astype(np.float32)
        weights = self.gen_trajectory_weights(traj, segment_edges).astype(np.float32)
        
        assert np.all(GT.shape < self.map_segment_size)
        
        pad_below = np.random.randint(self.map_segment_size - GT.shape)
        pad_above = self.map_segment_size - pad_below - GT.shape
        GT = np.pad(GT, ((pad_below[0],pad_above[0]), (pad_below[1], pad_above[1])), constant_values = -1)
        map_segment = np.pad(map_segment, ((pad_below[0],pad_above[0]), (pad_below[1], pad_above[1])), constant_values = -1)  

        assert np.all(weights >=0.), "pre padding"
        weights = np.pad(weights, ((pad_below[0],pad_above[0]), (pad_below[1], pad_above[1])), constant_values = 0)
        assert np.all(weights >=0.), "post padding"
        
        if self.rotate:
            n_rots = np.random.choice([0,1,2,3])
            angle = np.pi*n_rots/2
            rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

            GT = np.rot90(GT, n_rots)
            map_segment = np.rot90(map_segment, n_rots)
            weights = np.rot90(weights, n_rots)
            
            normalized_traj = (rot_mat@normalized_traj.T).T
        
        if self.transform:
            normalized_traj = self.augment_traj(normalized_traj)
            
        
        sample_dict = {"normalized input trajectory": normalized_traj.astype(np.float64),
                        "omap": map_segment.astype(np.float64),
                        "ground truth probability target": GT.astype(np.float64),
                        "ground truth trajectory": GT_traj.astype(np.float64),
                        "original predicted trajectory": traj.astype(np.float64),
                        'weights': weights.astype(np.float64)}
        return sample_dict
        
    def gaussian(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    def gen_ground_truth_image(self, traj, segment_edges, idx = 0):
        conv_res = self.mc.get_gt(traj)
        return self.cut_from_im(segment_edges, conv_res)

    def select_window_for_traj(self, traj):
        mins = np.min(traj, axis = 0)
        maxs = np.max(traj, axis = 0)
        if maxs[0] == mins[0]:
            maxs[0] += 1
        if maxs[1] == mins[1]:
            maxs[1] += 1

        return np.array([mins, maxs])
        
    def cut_from_im(self, segment_edges, im, cval = -1):
        segment_edges_local = np.copy(segment_edges)
        if segment_edges_local[0][0] < 0:
            im = np.pad(im, [(np.abs(segment_edges_local[0][0]), 0),(0,0)], constant_values = cval)
            segment_edges_local[1][0] += np.abs(segment_edges_local[0][0])
            segment_edges_local[0][0] = 0
            
        if segment_edges[0][1] < 0:
            im = np.pad(im, [(0,0),(np.abs(segment_edges_local[0][1]),0)], constant_values = cval)
            segment_edges_local[1][1] += np.abs(segment_edges_local[0][1])
            segment_edges_local[0][1] = 0
            
        if segment_edges_local[1][0] >= self.omap.shape[0]:
            im = np.pad(im, [(0,np.abs(segment_edges_local[1][0] - self.omap.shape[0])),(0,0)], constant_values = cval)
            
        if segment_edges_local[1][1] >= self.omap.shape[1]:
            im = np.pad(im, [(0,0),(0,np.abs(segment_edges_local[1][1] - self.omap.shape[1]))], constant_values = cval)
            
        segment = im[segment_edges_local[0][0]:segment_edges_local[1][0], 
                                segment_edges_local[0][1]:segment_edges_local[1][1]]

        return segment        
    
    def gen_map_segment(self, segment_edges):
        # cut out the part of the map where the trajectory is located pcut_from_im  padding
        cutout = self.cut_from_im(segment_edges, self.omap)
        return cutout
    
    
    def gen_trajectory_weights(self, traj, segment_edges):
        conv_res = self.mc.get_gt(traj)
        s = ndimage.generate_binary_structure(2,2)
        labeled_, n_areas = ndimage.label(conv_res > 0.25, structure = s)
        labeled = labeled_.copy().astype(float)
        labels = np.unique(labeled)
        for label in labels:
            if label == 0:
                continue
            n_pixels = float(np.count_nonzero(labeled == label))
            assert n_pixels > 0.
            inverter_val = np.count_nonzero(self.omap > 0.)
            labeled[labeled == label] = 0.000000001*np.exp(30*(inverter_val - n_pixels)/inverter_val)
            assert np.all(labeled >= 0.)
        if self.infeasible_space_weight is not None:
            labeled[labeled == 0] = self.infeasible_space_weight
        else:
            labeled[labeled == 0] = np.min(np.unique(labeled[labeled != 0])) ##TODO this might require adjustment to improve training, should this be a training param?
        cutout = self.cut_from_im(segment_edges, labeled, cval=0) #
        return cutout
        
    def augment_traj(self, normalized_traj):
        traj_odom = np.diff(normalized_traj, axis =0 ).astype(float)
        x_bias = np.random.rand() + 0.5
        traj_odom[:,0] = x_bias*traj_odom[:,0]
        y_bias = np.random.rand() + 0.5
        traj_odom[:,1] = y_bias*traj_odom[:,1]
        
        traj_odom = traj_odom + np.random.normal(scale = 0.25, size=traj_odom.shape)
        
        stz = np.vstack((np.array([[0,0]]), traj_odom))
        biased_traj = np.cumsum(stz, axis = 0).astype(int)
        return biased_traj


