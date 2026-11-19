from collections import defaultdict
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import cv2
import numpy as np
import openvino as ov
from openvino import opset1 as ops


def resize_rgb(image, height, width):
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def load_classifier_bundle_with_meta(bundle_path):
    core = ov.Core()
    with ZipFile(bundle_path) as bundle:
        names = set(bundle.namelist())
        labels = json.loads(bundle.read("classes.json"))
        metadata = json.loads(bundle.read("metadata.json")) if "metadata.json" in names else {}
        model = core.read_model(bundle.read("classifier.xml"), bundle.read("classifier.bin"))
    compiled = core.compile_model(
        model,
        "CPU",
        {"INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "LATENCY"},
    )
    return compiled, labels, metadata


def load_classifier_bundle(bundle_path):
    compiled, labels, _ = load_classifier_bundle_with_meta(bundle_path)
    return compiled, labels


def _output_shape(layer):
    return tuple(int(dim.text) for dim in layer.findall("output/port/dim"))


def _integers(attributes, name):
    return [int(value) for value in attributes[name].split(",")]


def _blob(layer, weights, name, dtype="<f4"):
    blob = layer.find(f"blobs/{name}")
    dtype = np.dtype(dtype)
    return np.frombuffer(
        weights,
        dtype=dtype,
        count=int(blob.attrib["size"]) // dtype.itemsize,
        offset=int(blob.attrib["offset"]),
    )


def _convolution(layer, attributes, image, weights):
    groups = int(attributes["group"])
    channels = image.get_output_shape(0)[1]
    filters = int(attributes["output"])
    kernel = _integers(attributes, "kernel")
    kernel_shape = (filters, channels // groups, *kernel)
    convolution = ops.convolution
    if groups > 1:
        kernel_shape = (groups, filters // groups, channels // groups, *kernel)
        convolution = ops.group_convolution

    result = convolution(
        image,
        _blob(layer, weights, "weights").reshape(kernel_shape),
        _integers(attributes, "strides"),
        _integers(attributes, "pads_begin"),
        _integers(attributes, "pads_end"),
        _integers(attributes, "dilations"),
        attributes.get("auto_pad", "explicit").upper(),
    )
    bias = _blob(layer, weights, "biases").reshape(1, filters, 1, 1)
    return ops.add(result, bias)


def _legacy_layer(layer, inputs, weights):
    kind = layer.attrib["type"]
    data = layer.find("data")
    attributes = data.attrib if data is not None else {}
    if kind == "Input":
        return ops.parameter(_output_shape(layer), dtype=np.float32)
    if kind == "Const":
        return ops.constant(_blob(layer, weights, "custom", "<i4"))
    if kind == "Convolution":
        return _convolution(layer, attributes, inputs[0], weights)
    if kind == "Clamp":
        return ops.clamp(inputs[0], float(attributes["min"]), float(attributes["max"]))
    if kind == "Reshape":
        return ops.reshape(inputs[0], inputs[1], special_zero=True)
    if kind == "Pooling" and attributes["pool-method"] == "avg":
        return ops.avg_pool(
            inputs[0],
            _integers(attributes, "strides"),
            _integers(attributes, "pads_begin"),
            _integers(attributes, "pads_end"),
            _integers(attributes, "kernel"),
            attributes.get("exclude-pad") == "true",
        )
    if kind == "FullyConnected":
        features = inputs[0].get_output_shape(0)[1]
        units = int(attributes["out-size"])
        matrix = _blob(layer, weights, "weights").reshape(units, features)
        bias = _blob(layer, weights, "biases")
        return ops.add(ops.matmul(inputs[0], matrix, False, True), bias)
    if kind == "SoftMax":
        return ops.softmax(inputs[0], int(attributes["axis"]))
    raise ValueError(f"The bundled model contains an unsupported layer ({kind}).")


def _read_legacy_model(graph, weights):
    incoming = defaultdict(list)
    used_as_input = set()
    for edge in graph.findall("edges/edge"):
        source = edge.attrib["from-layer"]
        target = edge.attrib["to-layer"]
        incoming[target].append((int(edge.attrib["to-port"]), source))
        used_as_input.add(source)

    nodes = {}
    parameters = []
    for layer in graph.findall("layers/layer"):
        layer_id = layer.attrib["id"]
        name = layer.attrib["name"]
        inputs = [nodes[source] for _, source in sorted(incoming[layer_id])]
        node = _legacy_layer(layer, inputs, weights)
        if tuple(node.get_output_shape(0)) != _output_shape(layer):
            raise ValueError(f"The bundled model has an unexpected output shape in {name}.")
        node.set_friendly_name(name)
        nodes[layer_id] = node
        if layer.attrib["type"] == "Input":
            parameters.append(node)

    outputs = [node for key, node in nodes.items() if key not in used_as_input]
    return ov.Model(outputs, parameters, graph.attrib["name"])


def compile_model(model_path):
    model_path = Path(model_path)
    core = ov.Core()
    graph = ElementTree.parse(model_path).getroot()
    if graph.attrib["version"] == "6":
        # The bundled 2019 model predates the IR versions supported by OpenVINO.
        model = _read_legacy_model(graph, model_path.with_suffix(".bin").read_bytes())
    else:
        model = core.read_model(model_path)
    return core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
