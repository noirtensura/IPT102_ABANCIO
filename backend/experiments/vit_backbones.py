"""Convert pretrained timm Vision Transformers into OpenVINO feature extractors.

The exported IR has the same contract as the Keras backbones in ``backbones.py`` (one NHWC RGB
float32 input in [0, 255], one pooled feature output), so feature caching, head training, and the
single-graph classifier export in ``backend/train.py`` work unchanged. Per-model normalisation is
baked into the graph.

Usage:
  .venv\\Scripts\\python.exe backend\\experiments\\vit_backbones.py build\\backbones dinov2_vits14_224
"""
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import timm
import torch

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    # name: timm checkpoint (all Apache-2.0), input size, pooling
    # "cls+mean" concatenates the class token with the mean patch token (DINOv2 linear-probe recipe).
    "dinov2_vits14_224": {"timm": "vit_small_patch14_dinov2.lvd142m", "size": 224, "pool": "cls+mean"},
    "dinov2_vitb14_224": {"timm": "vit_base_patch14_dinov2.lvd142m", "size": 224, "pool": "cls+mean"},
    "siglip2_vitb16_224": {"timm": "vit_base_patch16_siglip_224.v2_webli", "size": 224, "pool": "head"},
}


class FeatureExtractor(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, pool: str):
        super().__init__()
        self.model = model
        self.pool = pool
        self.prefix = int(getattr(model, "num_prefix_tokens", 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(x)
        if self.pool == "cls+mean":
            return torch.cat([tokens[:, 0], tokens[:, self.prefix :].mean(dim=1)], dim=1)
        return self.model.forward_head(tokens, pre_logits=True)


names = sys.argv[2:] or list(CANDIDATES)
core = ov.Core()
torch.manual_seed(0)
for name in names:
    spec = CANDIDATES[name]
    size = spec["size"]
    xml = OUT / f"{name}.xml"
    if not xml.is_file():
        base = timm.create_model(spec["timm"], pretrained=True, num_classes=0, img_size=size)
        mean = [255.0 * float(v) for v in base.pretrained_cfg["mean"]]
        std = [255.0 * float(v) for v in base.pretrained_cfg["std"]]
        wrapper = FeatureExtractor(base, spec["pool"]).eval()
        example = torch.zeros(1, 3, size, size)
        with torch.no_grad():
            converted = ov.convert_model(wrapper, example_input=example)
        converted.reshape(ov.PartialShape([-1, 3, size, size]))

        ppp = ov.preprocess.PrePostProcessor(converted)
        ppp.input().tensor().set_element_type(ov.Type.f32).set_layout(ov.Layout("NHWC"))
        ppp.input().model().set_layout(ov.Layout("NCHW"))
        ppp.input().preprocess().mean(mean).scale(std)
        converted = ppp.build()
        converted.input(0).get_tensor().set_names({"image"})
        # Write to a temporary name first so an interrupted export never leaves a half-written IR behind
        tmp_xml = OUT / f"{name}.partial.xml"
        ov.save_model(converted, tmp_xml, compress_to_fp16=False)
        tmp_xml.with_suffix(".bin").replace(xml.with_suffix(".bin"))
        tmp_xml.replace(xml)

        ref = np.random.default_rng(0).uniform(0, 255, (2, size, size, 3)).astype(np.float32)
        norm = (torch.from_numpy(ref) - torch.tensor(mean)) / torch.tensor(std)
        with torch.no_grad():
            expect = wrapper(norm.permute(0, 3, 1, 2)).numpy()
        got = core.compile_model(xml, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})(ref)[0]
        print(f"{name}: OV vs PyTorch max abs diff {np.abs(got - expect).max():.2e} (feature scale {np.abs(expect).mean():.2f})")
        del base, wrapper, converted

    net = core.compile_model(xml, "CPU", {"INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "THROUGHPUT"})
    batch = np.random.default_rng(1).uniform(0, 255, (16, size, size, 3)).astype(np.float32)
    net(batch)
    t = time.perf_counter()
    for _ in range(2):
        net(batch)
    bt = (time.perf_counter() - t) / 32
    lat = core.compile_model(xml, "CPU", {"INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "LATENCY"})
    one = batch[:1]
    lat(one)
    t = time.perf_counter()
    for _ in range(5):
        lat(one)
    lt = (time.perf_counter() - t) / 5
    print(
        f"{name}: dim={net.output(0).partial_shape[-1]} single-request batch {1 / bt:.1f} img/s, "
        f"single-image latency {lt * 1000:.0f} ms",
        flush=True,
    )
    del net, lat
