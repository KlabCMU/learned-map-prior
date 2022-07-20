import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.mobile_optimizer import optimize_for_mobile
import numpy as np
import random
from tqdm import tqdm
import logging

from .config import Config
from . import data
from .networks import MapPrior, MapPriorLightning, MapPriorTraj
from .train_test import train_loop

from .filter import filter
from .filter.utils import save_results, compute_errors


def train_lightning(config: Config):
    dataset = data.split(config.dataset_path / config.dataset)
    building = config.building
    files = dataset[building][config.train.split]

    random.shuffle(files)
    val_count = len(files) // 5
    train_files = files[:-val_count]
    val_files = files[:-val_count:]

    model = MapPriorLightning(config.data.sample_rate).double()

    dm = data.MapZoomedDataModule(
        train_building=building,
        train_files=train_files,
        val_files=val_files,
        samples_per_second=config.data.sample_rate,
        batch_size=config.train.batch_size,
        kernel_len=config.data.kernel_len,
        infeasible_space_weight=config.data.infeasbile_space_weight,
        transform=True,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=6,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        gpus=config.train.gpus,
        callbacks=[early_stop_callback],
        max_epochs=config.train.max_epochs
    )
    trainer.fit(model, dm)

    best_model_path = config.weights / \
        "{}_{}_best_weights.pth".format(
            config.dataset, config.building)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), best_model_path)

    logging.info("Generating map encoding")
    model.cuda().eval()
    test_traj = torch.zeros(
        config.data.sample_rate * config.data.kernel_len, 2).cuda().double()
    map = dm.train_map
    divisible = 2**3
    pad_x = divisible - map.shape[0] % divisible
    pad_y = divisible - map.shape[1] % divisible

    pad_x_sz_left = pad_x//2
    pad_x_sz_right = pad_x//2
    pad_y_sz_left = pad_y//2
    pad_y_sz_right = pad_y//2
    if pad_x % 2 == 1:
        pad_x_sz_left += 1
    if pad_y % 2 == 1:
        pad_y_sz_left += 1
    omap = np.pad(map, [(pad_x_sz_left, pad_x_sz_right),
                  (pad_y_sz_left, pad_y_sz_right)], constant_values=-1)
    test_map = torch.tensor(omap).cuda().double()

    net_output = model(test_traj[None, ...], test_map[None, ...])
    deep_map = net_output['deep map'].cpu().detach().numpy()[0]
    map_path = config.results / \
        config.map_filename.format(
            dataset=config.dataset, building=config.building)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(map_path, map=deep_map)
    generate_mobile(config)


def generate_mobile(config: Config):
    model = MapPriorTraj(config.data.sample_rate)
    model_dict = model.state_dict()

    best_model_path = config.weights / \
        "{}_{}_best_weights.pth".format(config.dataset, config.building)
    logging.info(f"Loading saved state {str(best_model_path)}")
    best_model_dict = torch.load(best_model_path)
    best_model_dict = {k:  v for k,
                       v in best_model_dict.items() if k in model_dict}
    model_dict.update(best_model_dict)
    model.load_state_dict(model_dict)

    logging.info("Scripting")
    mobile_model_path = config.results / \
        config.network_filename.format(
            dataset=config.dataset, building=config.building)
    traced_model = torch.jit.script(model)
    optimized = optimize_for_mobile(traced_model)
    optimized._save_for_lite_interpreter(str(mobile_model_path))


def run_filter(config: Config):
    odom_files = config.dataset_path.glob(
        f"{config.dataset}/{config.building}/test/*.npy")

    map_filename = (
        config.results
        / config.map_filename.format(dataset=config.dataset, building=config.building)
    )

    network_filename = (
        config.results
        / config.network_filename.format(dataset=config.dataset, building=config.building)
    )

    logging.info(f'Loading map embedding {map_filename}')
    deep_map = torch.from_numpy(np.load(map_filename)['map'])
    logging.info(f'Loading trajectory network {network_filename}')
    traj_net = torch.jit.load(network_filename)
    if not config.filter.allow_ble_update:
        logging.warn("Beacon updates not applied in filter!")

    for odom_file in odom_files:
        logging.info("Running PF on {}".format(odom_file))

        data = np.load(odom_file, allow_pickle=True).item()
        idol_preds = data['preds']
        ble_preds = data['BLE_preds'][config.data.num_beacons]
        gt_states = data['GT']
        idol_states = []
        ble_states = []
        filtered_states = []

        history_length = config.filter.history_length
        update_rate = config.filter.update_rate

        pf = filter.ParticleFilter(
            config,
            ble_preds[0],
            traj_net,
            deep_map
        )

        pf.init_particles()

        for idx, pred in enumerate(tqdm(idol_preds)):
            if idx == 0 or idx % config.filter.update_rate != 0:
                continue

            pf.predict(pred - idol_preds[idx - update_rate])
            if idx - update_rate * history_length >= 0:
                odom_history = idol_preds[idx - update_rate*history_length:idx]
                pf.update(odom_history,  ble_preds[idx//update_rate])

            if pf.reinit:
                logging.info(
                    "Most particles passed through wall, reinitializing...")
                pf.reinit_particles()

            pf.compute_state()
            pf.resample()
            filtered_states.append([pf.state[0], pf.state[1]])
            idol_states.append([pred[0], pred[1]])
            ble_states.append([ble_preds[idx//update_rate][0],
                              ble_preds[idx//update_rate][1]])

        filtered_states = np.array(filtered_states)
        idol_states = np.array(idol_states)
        ble_states = np.array(ble_states)
        save_results(gt_states, idol_states, ble_states,
                     filtered_states, config, odom_file)
    compute_errors(config)
