import numpy as np
from matplotlib import image
import os


def get_map(filename):
    im = image.imread(filename).T
    thresh = 0.99
    im[im < thresh] = 0.
    im[im >= thresh] = 1.

    im = 1-im.astype(int)

    # crop up to the edges of the buildings
    while np.all(im[0, :] == 0):
        im = im[1:, :]
    while np.all(im[-1, :] == 0):
        im = im[:-1, :]
    while np.all(im[:, 0] == 0):
        im = im[:, 1:]
    while np.all(im[:, -1] == 0):
        im = im[:, :-1]

    im = 1-im

    # blacken everything outside the actual map
    for idx in range(im.shape[0]):
        for jdx in range(im.shape[1]):
            if im[idx, jdx] != 0:
                im[idx, jdx] = 0
            else:
                break
        for jdx in range(im.shape[1]-1, 0, -1):
            if im[idx, jdx] != 0:
                im[idx, jdx] = 0
            else:
                break
    return im


def get_map_by_name(name):
    if name == 'building3':
        return get_map("/home/karnik/code/idol-map/dataset/maps/building3.png")
    elif name == 'building2_f1':
        return get_map("/home/karnik/code/idol-map/dataset/maps/building2.png")
    elif name == 'building2_f2':
        return get_map("/home/karnik/code/idol-map/dataset/maps/building2_f2_test.png")
    elif name == 'building1':
        # "building1_w_tables_smaller_smaller_tables.png"))#
        return get_map("/home/karnik/code/idol-map/dataset/maps/building1_map_more_tables.png")


def _world_to_image_coords(world_coords, map_size, world_size, origin):
    true_x_size = world_size[0]  # m
    true_y_size = world_size[1]  # m
    world_origin_offset_x = origin[0]  # m from top left corner of map
    world_origin_offset_y = origin[1]  # m from top left corner of map
    scale_factor_x = true_x_size/map_size[0]  # m/pixel
    scale_factor_y = true_y_size/map_size[1]  # m/pixel

    image_coord_x = np.round(
        (world_coords[:, 0] + world_origin_offset_x)/scale_factor_x).astype(int)
    image_coord_y = np.round(
        (world_coords[:, 1] + world_origin_offset_y)/scale_factor_y).astype(int)
    image_coords = np.hstack([image_coord_x[:, None], image_coord_y[:, None]])
    return image_coords


def _image_to_world_coords(image_coords, map_size, world_size, origin):
    true_x_size = world_size[0]  # m
    true_y_size = world_size[1]  # m
    world_origin_offset_x = origin[0]  # m from top left corner of map
    world_origin_offset_y = origin[1]  # m from top left corner of map
    scale_factor_x = true_x_size/map_size[0]  # m/pixel
    scale_factor_y = true_y_size/map_size[1]  # m/pixel

    world_coord_x = image_coords[:, 0]*scale_factor_x - world_origin_offset_x
    world_coord_y = image_coords[:, 1]*scale_factor_y - world_origin_offset_y
    world_coords = np.hstack([world_coord_x[:, None], world_coord_y[:, None]])

    return world_coords


_map_sizes = {
    "building3":  get_map_by_name("building3").shape,
    "building2_f1":   get_map_by_name("building2_f1").shape,
    "building2_f2":   get_map_by_name("building2_f2").shape,
    "building1": get_map_by_name("building1").shape,
}

_world_sizes = {
    "building3":  (121,    102.5),
    "building2_f1":   (69., 51),
    "building2_f2":   (67., 54.),
    "building1": (54, 18),
}

_origins = {  # world frame
    "building3":  (60.960, 10.973),
    "building2_f1":   (26.5,   34.5),
    "building2_f2":   (30.5,   29.),
    "building1": (14.023, 5.2),
}


def _coord_transform(world_coords_orig, map_name):
    if map_name == "building1":  # or map_name == 'building1_big':
        world_coords_new = world_coords_orig.copy()
        world_coords_new[:, 0] = -world_coords_orig[:, 0]
        world_coords_new[:, 1] = world_coords_orig[:, 1]
    elif map_name == "building3":
        world_coords_new = world_coords_orig.copy()
        world_coords_new[:, 1] = world_coords_orig[:, 0]
        world_coords_new[:, 0] = world_coords_orig[:, 1]
    elif map_name == "building2_f1":
        world_coords_new = world_coords_orig.copy()
        world_coords_new[:, 0] = world_coords_orig[:, 1]
        world_coords_new[:, 1] = world_coords_orig[:, 0]
    elif map_name == "building2_f2":
        world_coords_new = world_coords_orig.copy()
        world_coords_new[:, 0] = world_coords_orig[:, 1]
        world_coords_new[:, 1] = world_coords_orig[:, 0]
    return world_coords_new


def _inv_coord_transform(image_coords_orig, map_name):
    if map_name == "building1":  # or map_name == 'building1_big':
        image_coords_new = image_coords_orig.copy()
        image_coords_new[:, 0] = -image_coords_orig[:, 0]
        image_coords_new[:, 1] = image_coords_orig[:, 1]
    elif map_name == "building3":
        image_coords_new = image_coords_orig.copy()
        image_coords_new[:, 1] = image_coords_orig[:, 0]
        image_coords_new[:, 0] = image_coords_orig[:, 1]
    elif map_name == "building2_f1":
        image_coords_new = image_coords_orig.copy()
        image_coords_new[:, 0] = image_coords_orig[:, 1]
        image_coords_new[:, 1] = image_coords_orig[:, 0]
    elif map_name == "building2_f2":
        image_coords_new = image_coords_orig.copy()
        image_coords_new[:, 0] = image_coords_orig[:, 1]
        image_coords_new[:, 1] = image_coords_orig[:, 0]
    return image_coords_new


def world_to_image_coords(world_coords, map_name):
    return _world_to_image_coords(_coord_transform(world_coords, map_name), _map_sizes[map_name], _world_sizes[map_name], _origins[map_name])


def image_to_world_coords(image_coords, map_name):
    return _inv_coord_transform(_image_to_world_coords(image_coords, _map_sizes[map_name], _world_sizes[map_name], _origins[map_name]), map_name)
