### Setup
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
