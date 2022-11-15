## Learnable Spatio-Temporal Map Embeddings for Deep Inertial Localization
Dennis Melamed, Karnik Ram, Vivek Roy, Kris Kitani.

[[`arXiv`](https://arxiv.org/pdf/2211.07635)]
[[`Project Page`](https://klabcmu.github.io/learned-map-prior/)]

In IROS 2022

<p align="center">
<a href="https://github.com/KlabCMU/learned-map-prior/blob/ed715d6b21ecbce20e12f01bddc83a4f7b237132/resources/teaser.png"><img src="https://github.com/KlabCMU/learned-map-prior/blob/ed715d6b21ecbce20e12f01bddc83a4f7b237132/resources/teaser.png" width="700"/></a>
</p>
Our proposed method utilizes learnable spatio-temporal map priors to reduce drift in inertial odometry.

## Setup
```bash
conda env create --file map-prior.yml
```

### Train
```bash
python main.py --mode TrainLightning --dataset BLE_IMU --building building2_f1 --train-gpus 2 --data-sample-rate 60
python main.py --mode TrainLightning --dataset IDOL --building building2_f1 --train-gpus 2 --data-sample-rate 100
```

### Filter
```bash
python main.py --mode RunFilter --dataset BLE_IMU --building building2_f1
python main.py --mode RunFilter --dataset BLE_IMU --building building2_f1 --no-filter-allow-ble-update
python main.py --mode RunFilter --dataset IDOL --building building1 --filter-update-rate 100
python main.py --mode RunFilter --dataset IDOL --building building1 --filter-update-rate 100 --no-filter-allow-reinit
```
