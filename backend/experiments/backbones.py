"""Convert candidate Keras backbones to OpenVINO feature extractors and benchmark CPU speed."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import openvino as ov
import tensorflow as tf
from tensorflow import keras

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    # name: (constructor, input size, needs [-1,1] rescale)
    "mobilenetv2_035_128": (lambda s: keras.applications.MobileNetV2(input_shape=(s, s, 3), alpha=0.35, include_top=False, weights="imagenet", pooling="avg"), 128, True),
    "effnetv2b0_224": (lambda s: keras.applications.EfficientNetV2B0(input_shape=(s, s, 3), include_top=False, weights="imagenet", pooling="avg"), 224, False),
    "effnetv2b2_260": (lambda s: keras.applications.EfficientNetV2B2(input_shape=(s, s, 3), include_top=False, weights="imagenet", pooling="avg"), 260, False),
    "convnext_tiny_224": (lambda s: keras.applications.ConvNeXtTiny(input_shape=(s, s, 3), include_top=False, weights="imagenet", pooling="avg"), 224, False),
}

names = sys.argv[2:] or list(CANDIDATES)
core = ov.Core()
for name in names:
    xml = OUT / f"{name}.xml"
    build, size, rescale = CANDIDATES[name]
    if not xml.is_file():
        inp = keras.Input((size, size, 3), name="image")
        base = build(size)
        x = keras.layers.Rescaling(1 / 127.5, offset=-1)(inp) if rescale else inp
        model = keras.Model(inp, base(x, training=False))
        sm = OUT / f"{name}_sm"
        model.export(sm, input_signature=[tf.TensorSpec((None, size, size, 3), tf.float32, name="image")], verbose=False)
        ov.save_model(ov.convert_model(sm), xml, compress_to_fp16=False)
        ref = np.random.default_rng(0).uniform(0, 255, (2, size, size, 3)).astype(np.float32)
        expect = model(ref, training=False).numpy()
        got = core.compile_model(xml, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})(ref)[0]
        print(f"{name}: OV vs Keras max abs diff {np.abs(got - expect).max():.2e}")
    net = core.compile_model(xml, "CPU", {"INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "THROUGHPUT"})
    batch = np.random.default_rng(1).uniform(0, 255, (16, size, size, 3)).astype(np.float32)
    net(batch)
    t = time.perf_counter(); [net(batch) for _ in range(3)]; bt = (time.perf_counter() - t) / 48
    lat = core.compile_model(xml, "CPU", {"INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "LATENCY"})
    one = batch[:1]; lat(one)
    t = time.perf_counter(); [lat(one) for _ in range(10)]; lt = (time.perf_counter() - t) / 10
    print(f"{name}: dim={net.output(0).partial_shape[-1]} batch {1/bt:.1f} img/s, single-image latency {lt*1000:.0f} ms", flush=True)
