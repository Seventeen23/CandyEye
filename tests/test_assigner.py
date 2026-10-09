"""Task-aligned assigner conflict resolution and target quality."""
import numpy as np
import torch

from candyeye.training.assigner import TaskAlignedAssigner


def _grid():
    xs = torch.arange(4) * 8 + 4.0
    return torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).view(-1, 2)


def test_conflict_resolution_keeps_anchor_inside_assigned_target():
    assigner = TaskAlignedAssigner(topk=10, alpha=.5, beta=6.0)
    grid = _grid()
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(60):
        n = int(rng.integers(2, 5))
        x1 = torch.tensor(rng.uniform(0, 40, (n, 1)), dtype=torch.float32)
        y1 = torch.tensor(rng.uniform(0, 40, (n, 1)), dtype=torch.float32)
        boxes = torch.cat([
            x1, y1,
            x1 + torch.tensor(rng.uniform(6, 30, (n, 1)), dtype=torch.float32),
            y1 + torch.tensor(rng.uniform(6, 30, (n, 1)), dtype=torch.float32),
        ], dim=1).unsqueeze(0)
        # Large predicted boxes give every GT a chance to win the old,
        # unmasked argmax over all GTs (including non-claimants).
        pred = torch.stack((grid[:, 0] - 14, grid[:, 1] - 14,
                            grid[:, 0] + 14, grid[:, 1] + 14), dim=1).unsqueeze(0)
        out = assigner(torch.full((1, 16, n), 0.9), pred, grid,
                       torch.arange(n).unsqueeze(0), boxes,
                       torch.ones(1, n, dtype=torch.bool))
        for anchor in out["fg_mask"][0].nonzero().flatten().tolist():
            checked += 1
            target = out["boxes"][0, anchor]
            cx, cy = float(grid[anchor, 0]), float(grid[anchor, 1])
            assert target[0] < cx < target[2], "anchor x outside assigned target"
            assert target[1] < cy < target[3], "anchor y outside assigned target"
    assert checked > 0


def test_target_quality_comes_from_the_assigned_gt():
    # Anchor 0 is claimed only by the small GT (the big GT ranks it outside
    # its top-10). The unmasked quality max let the *unclaimed* big GT's
    # alignment (~0.9) win and be written to the small GT's class slot.
    assigner = TaskAlignedAssigner(topk=10, alpha=.5, beta=6.0)
    grid = _grid()
    pred = torch.stack((grid[:, 0] - 30, grid[:, 1] - 30,
                        grid[:, 0] + 30, grid[:, 1] + 30), dim=1).unsqueeze(0)
    gt_boxes = torch.tensor([[[0., 0., 8., 8.], [0., 0., 64., 64.]]])
    scores = torch.full((1, 16, 2), 0.9)
    scores[0, 0, 1] = 0.85

    out = assigner(scores, pred, grid, torch.tensor([[0, 1]]), gt_boxes,
                   torch.ones(1, 2, dtype=torch.bool))

    assert bool(out["fg_mask"][0, 0])
    label = int(out["labels"][0, 0])
    quality = float(out["scores"][0, 0, label])
    assert label == 0
    # Quality must be the (clamped) alignment of GT0, not ~0.9 from GT1.
    assert quality <= 0.2501
