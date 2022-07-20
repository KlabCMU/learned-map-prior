import torch
from tqdm import tqdm


def loss(maps, y, traj_embeddings, wgt=None):
    se = torch.pow(maps - y, 2)

    if wgt is not None:
        se = wgt*se
        assert torch.all(se >= 0)
        assert torch.all(wgt >= 0)
    mse = torch.mean(se)
    return mse


def train_loop(model,
               trainData,
               valData,
               optimizer,
               epochs,
               device=None):
    if device is not None:
        model = model.to(device)
    bestState = None
    bestLoss = float('inf')
    for epoch in range(epochs):
        trainLoss = 0
        valLoss = 0

        model.train()
        batch = tqdm(trainData, leave=False, position=0)
        for i, data in enumerate(batch):
            input_traj = data["normalized input trajectory"]
            map = data["omap"]
            target_scores = data["ground truth probability target"]
            if device is not None:
                input_traj, map, target_scores = input_traj.to(
                    device), map.to(device), target_scores.to(device)
            input_traj, map = input_traj.float(), map.float()
            out = model(input_traj, map)
            predicted_scores = out['convolution result']
            l = loss(predicted_scores, target_scores,
                     out['deep traj vector'], wgt=data['weights'].to(device))
            trainLoss = l if i == 0 else trainLoss * (i /
                                                      (i + 1)) + l / (i + 1)
            batch.set_postfix(loss=f"{trainLoss:8.4f}")
            optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        with torch.no_grad():
            batch = tqdm(valData, leave=False, position=0)
            for i, data in enumerate(batch):
                input_traj = data["normalized input trajectory"]
                map = data["omap"]
                target_scores = data["ground truth probability target"]
                if device is not None:
                    input_traj, map, target_scores = input_traj.to(
                        device), map.to(device), target_scores.to(device)
                input_traj, map = input_traj.float(), map.float()
                out = model(input_traj, map)
                predicted_scores = out['convolution result']
                l = loss(predicted_scores, target_scores,
                         out['deep traj vector'], wgt=data['weights'].to(device))
                valLoss = l if i == 0 else valLoss * (i /
                                                      (i + 1)) + l / (i + 1)
                batch.set_postfix(loss=f"{valLoss:8.4f}")

        tqdm.write(
            f"Epoch [{epoch+1}/{epochs}], Train: {trainLoss:8.4f}, Val: {valLoss: 8.4f}"
        )
        if valLoss < bestLoss:
            bestState = model.state_dict()
            bestLoss = valLoss

    return bestState, bestLoss


def test_loop():
    pass
