"""Hand-computed tests for VOC AP and difficult annotations."""
import pytest

from eval.map import evaluate_map, evaluate_map50


def test_all_point_ap_matches_hand_computed_example():
    gt = {
        "img1": {"boxes": [[0, 0, 10, 10]], "labels": [0], "difficult": [False]},
        "img2": {"boxes": [[20, 20, 30, 30]], "labels": [0], "difficult": [False]},
    }
    detections = [
        {"image_id": "img1", "class_id": 0, "score": .9, "box": [0, 0, 10, 10]},
        {"image_id": "img1", "class_id": 0, "score": .8, "box": [40, 40, 50, 50]},
        {"image_id": "img2", "class_id": 0, "score": .7, "box": [20, 20, 30, 30]},
    ]

    mean_ap, aps = evaluate_map50(gt, detections, num_classes=2)

    assert aps[0] == pytest.approx(5 / 6)
    assert aps[1] is None
    assert mean_ap == pytest.approx(5 / 6)


def test_detection_on_difficult_object_is_ignored():
    gt = {"image": {
        "boxes": [[0, 0, 10, 10], [20, 20, 30, 30]],
        "labels": [0, 0], "difficult": [False, True],
    }}
    detections = [
        {"image_id": "image", "class_id": 0, "score": .9, "box": [0, 0, 10, 10]},
        {"image_id": "image", "class_id": 0, "score": .8, "box": [20, 20, 30, 30]},
        {"image_id": "image", "class_id": 0, "score": .7, "box": [40, 40, 50, 50]},
    ]

    mean_ap, aps = evaluate_map50(gt, detections, num_classes=1)

    assert aps[0] == pytest.approx(1.0)
    assert mean_ap == pytest.approx(1.0)


def test_out_of_range_detection_class_raises():
    gt = {"image": {"boxes": [[0, 0, 10, 10]], "labels": [0]}}
    detections = [
        {"image_id": "image", "class_id": 3, "score": .9, "box": [0, 0, 10, 10]},
    ]

    with pytest.raises(ValueError, match="num_classes mismatch"):
        evaluate_map50(gt, detections, num_classes=2)


def test_out_of_range_gt_label_raises():
    gt = {"image": {"boxes": [[0, 0, 10, 10]], "labels": [7]}}

    with pytest.raises(ValueError, match="class ids"):
        evaluate_map50(gt, [], num_classes=2)


def test_coco_map_averages_iou_sweep():
    gt = {"img": {"boxes": [[0, 0, 10, 10]], "labels": [0], "difficult": [False]}}
    detections = [{"image_id": "img", "class_id": 0, "score": .9, "box": [0, 0, 10, 10]}]

    result = evaluate_map(gt, detections, num_classes=1)

    assert result["map"] == pytest.approx(1.0)
    assert result["map50"] == pytest.approx(1.0)
    assert result["map75"] == pytest.approx(1.0)
    assert len(result["iou_thresholds"]) == 10


def test_coco_map_drops_when_iou_does_not_hold_up():
    gt = {"img": {"boxes": [[0, 0, 10, 10]], "labels": [0], "difficult": [False]}}
    # IoU ~= 0.83: matched at 0.5..0.8 but not at 0.85+.
    detections = [{"image_id": "img", "class_id": 0, "score": .9, "box": [0, 0, 12, 10]}]

    result = evaluate_map(gt, detections, num_classes=1)

    assert result["map50"] == pytest.approx(1.0)
    assert result["map"] < result["map50"]


def test_size_buckets_are_assigned_by_area():
    gt = {"img": {
        "boxes": [[0, 0, 4, 4], [0, 0, 30, 30]],
        "labels": [0, 0], "difficult": [False, False],
    }}
    detections = [
        {"image_id": "img", "class_id": 0, "score": .9, "box": [0, 0, 4, 4]},
        {"image_id": "img", "class_id": 0, "score": .8, "box": [0, 0, 30, 30]},
    ]

    result = evaluate_map(gt, detections, num_classes=1, image_size=128,
                          size_buckets=True)

    assert result["size"]["small"] == pytest.approx(1.0)
    assert result["size"]["large"] == pytest.approx(1.0)
    assert result["size"]["medium"] is None

