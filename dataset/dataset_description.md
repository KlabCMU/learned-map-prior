Each dataset is divided into the buildings it was collected in.

Each building is then divided into train and test data.

For IDOL dataset, training files contain only a "GT" entry, testing files contain "GT" and "preds" entries
GT is 2D position data, preds is predicted odometry positions (2D)


For BLE+IMU dataset, training files again contain only "GT" entries, testing files contain "GT", "preds", "BLE_preds"
BLE_preds are the results of ble localization and is a dictionary. The dictionary key is the number of beacons used to generate those results, and the value is the actual 2D predictions.
We do not train on building2 f1 again, since we can reuse the idol trained data, we only retrain the map prior for building2 f2

For Robot dataset, same as before but "preds" are wheel encoder odometry predictions, and we do not have a train set since the odometry method does not need it and the map prior can generalize from inertial odom training data.


