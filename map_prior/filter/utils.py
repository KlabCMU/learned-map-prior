import numpy as np
from ..map import processing
import matplotlib.pyplot as plt
from scipy import ndimage
import pickle
import logging


def sample(odometry_cov):
    return np.random.multivariate_normal(np.zeros((odometry_cov.shape[0],)), odometry_cov)


def normpdf(x, mean, std):
    var = std**2
    denom = np.sqrt(2*np.pi*var)
    num = np.exp((-(x - mean)**2)/(2*var))
    return num/denom


def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end

    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    Source: http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else [x, y]
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return np.array(points)


def passed_through_wall(point_0, point_1, map, building):
    point_0_im = processing.world_to_image_coords(
        point_0[np.newaxis, ...], building)[0]
    point_1_im = processing.world_to_image_coords(
        point_1[np.newaxis, ...], building)[0]
    passed = False
    line_points = get_line(point_0_im, point_1_im)
    if point_1_im[0] >= map.shape[0] or point_1_im[1] >= map.shape[1] or point_1_im[0] < 0 or point_1_im[1] < 0:  # out of bounds
        passed = True
    elif np.any(line_points[:, 0] >= map.shape[0]) or np.any(line_points[:, 1] >= map.shape[1]) or np.any(line_points[:, 0] < 0) or np.any(line_points[:, 1] < 0):  # path is out of bounds
        passed = True
    elif np.any(map[line_points[1:, 0], line_points[1:, 1]] == 0.):  # passed through a wall
        passed = True
    return passed


def save_results(gt_states, idol_states, ble_states, filtered_states, config, odom_file):

    results_dict = {}
    results_dict['gt'] = gt_states
    results_dict['idol'] = idol_states
    results_dict['ble'] = ble_states
    results_dict['pf'] = filtered_states
    results_save_path = config.results / \
        f"{config.dataset}_{config.building}_{odom_file.stem}.pkl"
    pickle.dump(results_dict, open(results_save_path, 'wb'))

    gt_states_px = processing.world_to_image_coords(gt_states, config.building)
    idol_states_px = processing.world_to_image_coords(
        idol_states, config.building)
    ble_states_px = processing.world_to_image_coords(
        ble_states, config.building)
    filtered_states_px = processing.world_to_image_coords(
        filtered_states, config.building)

    map = ndimage.binary_dilation(processing.get_map_by_name(config.building))

    plt.figure()
    plt.imshow(map.T, cmap='gray')
    plt.plot(gt_states_px[:, 0], gt_states_px[:, 1], c='k', label='gt')
    plt.plot(idol_states_px[:, 0], idol_states_px[:, 1], c='b', label='idol')
    plt.plot(ble_states_px[:, 0], ble_states_px[:, 1], c='g', label='ble')
    plt.plot(filtered_states_px[:, 0],
             filtered_states_px[:, 1], c='r', label='filtered')
    plt.legend()
    fig_save_path = config.results / \
        f"{config.dataset}_{config.building}_{odom_file.stem}.png"
    plt.savefig(fig_save_path)


def compute_errors(config):
    result_files = config.results.glob(
        f'{config.dataset}_{config.building}_*.pkl')

    errors = {"ATE": {"IDOL": [], "BLE": [], "PF": []}}
    for file in result_files:
        results_dict = pickle.load(open(file, 'rb'))
        update_rate = config.filter.update_rate
        gt = results_dict['gt'][:-update_rate:update_rate]
        idol = results_dict['idol']
        ble = results_dict['ble']
        pf = results_dict['pf']

        idol_error = np.linalg.norm(gt - idol, axis=1)
        errors["ATE"]["IDOL"].append(idol_error)

        ble_error = np.linalg.norm(gt - ble, axis=1)
        errors["ATE"]["BLE"].append(ble_error)

        pf_error = np.linalg.norm(gt - pf, axis=1)
        errors["ATE"]["PF"].append(pf_error)

        logging.info(
            f"File: {file}, IDOL-ATE: {np.sqrt(np.mean((idol_error)**2))}, BLE-ATE: {np.sqrt(np.mean((ble_error)**2))}, PF-ATE: {np.sqrt(np.mean((pf_error)**2))}"
        )

    for key in ["IDOL", "BLE", "PF"]:
        errors["ATE"][key] = np.sqrt(np.mean(np.hstack(errors["ATE"][key])**2))

    print("Overall:", errors)
