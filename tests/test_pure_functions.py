"""Tests for pure helper functions across the benchmark scripts."""

import pytest

from benchmarks.run_benchmark import _percentile
from benchmarks.compare_models import short_name
from app.batch_inference import decode_ssd


# ---------- _percentile ----------

def test_percentile_empty():
    assert _percentile([], 50) == 0.0

def test_percentile_p0_is_min():
    data = [3.0, 1.0, 2.0, 5.0, 4.0]
    assert _percentile(sorted(data), 0) == 1.0

def test_percentile_p100_is_max():
    data = [3.0, 1.0, 2.0, 5.0, 4.0]
    assert _percentile(sorted(data), 100) == 5.0

def test_percentile_median_odd():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(sorted(data), 50) == 3.0

def test_percentile_interpolates():
    # 0,10,20,30 -> p50 should be 15.0
    data = [0.0, 10.0, 20.0, 30.0]
    assert _percentile(sorted(data), 50) == 15.0


# ---------- short_name ----------

def test_short_name_face_detection():
    assert short_name("face-detection-retail-0004.xml") == "face-detection"

def test_short_name_mobilenet_fp16():
    assert short_name("mobilenet-v3-large-1.0-224-tf.xml") == "mobilenet-v3-large"

def test_short_name_mobilenet_int8():
    # INT8 suffix must be preserved so the comparison table can distinguish
    assert short_name("mobilenet-v3-large-1.0-224-tf-int8.xml") == "mobilenet-v3-large-int8"


# ---------- decode_ssd ----------

def _make_output(rows):
    """Build a fake [1,1,N,7] SSD output tensor."""
    import numpy as np
    return np.array(rows, dtype=np.float32).reshape(1, 1, len(rows), 7)

def test_decode_ssd_filters_by_confidence():
    rows = [
        [0, 1, 0.9, 0.1, 0.2, 0.3, 0.4],
        [0, 1, 0.3, 0.5, 0.6, 0.7, 0.8],   # below threshold
    ]
    boxes = decode_ssd(_make_output(rows), conf_thresh=0.5)
    assert len(boxes) == 1
    assert boxes[0][4] == pytest.approx(0.9)

def test_decode_ssd_clamps_coordinates():
    rows = [
        [0, 1, 0.9, -0.5, -0.2, 1.5, 1.2],  # out of range
    ]
    boxes = decode_ssd(_make_output(rows), conf_thresh=0.5)
    assert len(boxes) == 1
    xmin, ymin, xmax, ymax, _conf = boxes[0]
    assert xmin == 0.0
    assert ymin == 0.0
    assert xmax == 1.0
    assert ymax == 1.0

def test_decode_ssd_keeps_box_order():
    rows = [
        [0, 1, 0.9, 0.1, 0.2, 0.3, 0.4],
        [0, 1, 0.8, 0.5, 0.6, 0.7, 0.8],
    ]
    boxes = decode_ssd(_make_output(rows), conf_thresh=0.5)
    assert len(boxes) == 2
    assert boxes[0][4] == pytest.approx(0.9)
    assert boxes[1][4] == pytest.approx(0.8)
