"""Export the SAMA network to ONNX (fp32, fp16-weights, int8-weights) without the `onnx` package.

The ONNX protobuf messages are built from a hand-written subset of onnx.proto (field numbers as
in the ONNX specification, IR version 8, opset 13) with google.protobuf, then validated and run
with onnxruntime (tests/test_export.py). fp16 and int8 are weight-storage formats: weights are
stored in half precision / int8 with a per-output-channel scale and cast or dequantised to
float32 inside the graph (DequantizeLinear), so compute stays fp32 on CPU.
"""
from __future__ import annotations

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

_F = descriptor_pb2.FieldDescriptorProto


def _build_classes():
    fd = descriptor_pb2.FileDescriptorProto(name="onnx_subset.proto", package="onnx", syntax="proto2")

    def msg(name, fields, nested=()):
        m = fd.message_type.add(name=name)
        for fname, num, ftype, label, tname in fields:
            f = m.field.add(name=fname, number=num, type=ftype, label=label)
            if tname:
                f.type_name = tname
        for n in nested:
            m.nested_type.add().CopyFrom(n)
        return m

    O, R = _F.LABEL_OPTIONAL, _F.LABEL_REPEATED
    msg("TensorProto", [("dims", 1, _F.TYPE_INT64, R, None), ("data_type", 2, _F.TYPE_INT32, O, None),
                        ("float_data", 4, _F.TYPE_FLOAT, R, None), ("int64_data", 7, _F.TYPE_INT64, R, None),
                        ("name", 8, _F.TYPE_STRING, O, None), ("raw_data", 9, _F.TYPE_BYTES, O, None)])
    msg("AttributeProto", [("name", 1, _F.TYPE_STRING, O, None), ("f", 2, _F.TYPE_FLOAT, O, None),
                           ("i", 3, _F.TYPE_INT64, O, None), ("s", 4, _F.TYPE_BYTES, O, None),
                           ("t", 5, _F.TYPE_MESSAGE, O, ".onnx.TensorProto"),
                           ("floats", 7, _F.TYPE_FLOAT, R, None), ("ints", 8, _F.TYPE_INT64, R, None),
                           ("type", 20, _F.TYPE_INT32, O, None)])
    dim = descriptor_pb2.DescriptorProto(name="Dimension")
    dim.field.add(name="dim_value", number=1, type=_F.TYPE_INT64, label=O)
    dim.field.add(name="dim_param", number=2, type=_F.TYPE_STRING, label=O)
    msg("TensorShapeProto", [("dim", 1, _F.TYPE_MESSAGE, R, ".onnx.TensorShapeProto.Dimension")], nested=[dim])
    tens = descriptor_pb2.DescriptorProto(name="Tensor")
    tens.field.add(name="elem_type", number=1, type=_F.TYPE_INT32, label=O)
    tens.field.add(name="shape", number=2, type=_F.TYPE_MESSAGE, label=O, type_name=".onnx.TensorShapeProto")
    msg("TypeProto", [("tensor_type", 1, _F.TYPE_MESSAGE, O, ".onnx.TypeProto.Tensor")], nested=[tens])
    msg("ValueInfoProto", [("name", 1, _F.TYPE_STRING, O, None), ("type", 2, _F.TYPE_MESSAGE, O, ".onnx.TypeProto")])
    msg("NodeProto", [("input", 1, _F.TYPE_STRING, R, None), ("output", 2, _F.TYPE_STRING, R, None),
                      ("name", 3, _F.TYPE_STRING, O, None), ("op_type", 4, _F.TYPE_STRING, O, None),
                      ("attribute", 5, _F.TYPE_MESSAGE, R, ".onnx.AttributeProto"), ("domain", 7, _F.TYPE_STRING, O, None)])
    msg("GraphProto", [("node", 1, _F.TYPE_MESSAGE, R, ".onnx.NodeProto"), ("name", 2, _F.TYPE_STRING, O, None),
                       ("initializer", 5, _F.TYPE_MESSAGE, R, ".onnx.TensorProto"),
                       ("doc_string", 10, _F.TYPE_STRING, O, None),
                       ("input", 11, _F.TYPE_MESSAGE, R, ".onnx.ValueInfoProto"),
                       ("output", 12, _F.TYPE_MESSAGE, R, ".onnx.ValueInfoProto")])
    msg("OperatorSetIdProto", [("domain", 1, _F.TYPE_STRING, O, None), ("version", 2, _F.TYPE_INT64, O, None)])
    msg("ModelProto", [("ir_version", 1, _F.TYPE_INT64, O, None), ("producer_name", 2, _F.TYPE_STRING, O, None),
                       ("producer_version", 3, _F.TYPE_STRING, O, None), ("doc_string", 6, _F.TYPE_STRING, O, None),
                       ("graph", 7, _F.TYPE_MESSAGE, O, ".onnx.GraphProto"),
                       ("opset_import", 8, _F.TYPE_MESSAGE, R, ".onnx.OperatorSetIdProto")])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    return {n: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"onnx.{n}"))
            for n in ("TensorProto", "AttributeProto", "ValueInfoProto", "NodeProto", "GraphProto", "ModelProto", "OperatorSetIdProto")}


PB = _build_classes()
FLOAT, INT8, INT64, FLOAT16 = 1, 3, 7, 10
_NP = {FLOAT: np.float32, INT8: np.int8, INT64: np.int64, FLOAT16: np.float16}


def tensor(name, arr, dtype=FLOAT):
    a = np.ascontiguousarray(np.asarray(arr).astype(_NP[dtype]))
    t = PB["TensorProto"](name=name, data_type=dtype, raw_data=a.tobytes())
    t.dims.extend(a.shape)
    return t


def attr(name, v):
    A = PB["AttributeProto"](name=name)
    if isinstance(v, (list, tuple)):
        A.ints.extend(int(x) for x in v)
        A.type = 7
    elif isinstance(v, float):
        A.f, A.type = v, 1
    else:
        A.i, A.type = int(v), 2
    return A


def node(op, inputs, outputs, **attrs):
    n = PB["NodeProto"](op_type=op, name=f"{op}_{outputs[0]}")
    n.input.extend(inputs)
    n.output.extend(outputs)
    for k, v in attrs.items():
        n.attribute.append(attr(k, v))
    return n


def value_info(name, shape, elem=FLOAT):
    vi = PB["ValueInfoProto"](name=name)
    vi.type.tensor_type.elem_type = elem
    for d in shape:
        dd = vi.type.tensor_type.shape.dim.add()
        if isinstance(d, str):
            dd.dim_param = d
        else:
            dd.dim_value = d
    return vi


def build_model(P: dict, T: int = 450, weights: str = "fp32", meta: str = "") -> bytes:
    """Build the ONNX model of sama.net.forward. weights: fp32 | fp16 | int8."""
    nodes, inits = [], []

    def W(name, arr, per_channel_axis=0):
        """Add a weight; returns the float32 tensor name to use in the graph."""
        arr = np.asarray(arr, np.float32)
        if weights == "fp32":
            inits.append(tensor(name, arr))
            return name
        if weights == "fp16":
            inits.append(tensor(name + "_h", arr, FLOAT16))
            nodes.append(node("Cast", [name + "_h"], [name], to=FLOAT))
            return name
        # int8: symmetric per-output-channel (axis 0) quantisation for >=2D, per-tensor for 1D
        if arr.ndim >= 2:
            amax = np.abs(arr).reshape(arr.shape[0], -1).max(1)
            scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
            q = np.clip(np.round(arr / scale.reshape((-1,) + (1,) * (arr.ndim - 1))), -127, 127)
            inits.extend([tensor(name + "_q", q, INT8), tensor(name + "_s", scale), tensor(name + "_z", np.zeros(arr.shape[0]), INT8)])
            nodes.append(node("DequantizeLinear", [name + "_q", name + "_s", name + "_z"], [name], axis=per_channel_axis))
            return name
        inits.append(tensor(name, arr))  # biases stay fp32
        return name

    C = P["W1"].shape[0]
    inits.append(tensor("ax2", np.array([2]), INT64))
    inits.append(tensor("ax1", np.array([1]), INT64))
    inits.append(tensor("s0", np.array([0]), INT64))
    inits.append(tensor("s1", np.array([1]), INT64))
    inits.append(tensor("onesC", np.ones(C)))
    wf = W("Wf", P["Wf"])
    inits.append(tensor("bf", P["bf"]))
    nodes += [node("Gemm", ["cond", wf, "bf"], ["hpre"], transB=1), node("Tanh", ["hpre"], ["h"])]
    for l in (1, 2):
        wg, wb = W(f"Wg{l}", P[f"Wg{l}"]), W(f"Wb{l}", P[f"Wb{l}"])
        inits.append(tensor(f"zerosC{l}", np.zeros(C)))
        nodes += [node("Gemm", ["h", wg, "onesC"], [f"g{l}"], transB=1),
                  node("Gemm", ["h", wb, f"zerosC{l}"], [f"be{l}"], transB=1),
                  node("Unsqueeze", [f"g{l}", "ax2"], [f"g{l}u"]), node("Unsqueeze", [f"be{l}", "ax2"], [f"be{l}u"])]
    inp = "x"
    for l, d in ((1, 1), (2, 2), (3, 4), (4, 1)):
        k = P[f"W{l}"].shape[2]
        w = W(f"W{l}", P[f"W{l}"])
        inits.append(tensor(f"b{l}", P[f"b{l}"]))
        pad = d * (k - 1) // 2
        nodes.append(node("Conv", [inp, w, f"b{l}"], [f"z{l}"], dilations=[d], kernel_shape=[k], pads=[pad, pad]))
        cur = f"z{l}"
        if l in (1, 2):
            nodes += [node("Mul", [cur, f"g{l}u"], [f"m{l}"]), node("Add", [f"m{l}", f"be{l}u"], [f"f{l}"])]
            cur = f"f{l}"
        if l in (1, 2, 3):
            nodes.append(node("Relu", [cur], [f"a{l}"]))
            inp = f"a{l}"
    nodes += [node("Slice", ["x", "s0", "s1", "ax1"], ["x0"]), node("Add", ["x0", "z4"], ["ysum"]),
              node("Squeeze", ["ysum", "ax1"], ["y"])]
    g = PB["GraphProto"](name="sama_net", doc_string=meta)
    g.node.extend(nodes)
    g.initializer.extend(inits)
    g.input.extend([value_info("x", ["N", 4, T]), value_info("cond", ["N", 2])])
    g.output.extend([value_info("y", ["N", T])])
    m = PB["ModelProto"](ir_version=8, producer_name="sama-insilico", producer_version="1.0", graph=g,
                         doc_string="SAMA step-4 network (research use only, not a medical device)")
    m.opset_import.append(PB["OperatorSetIdProto"](domain="", version=13))
    return m.SerializeToString()


def save_all(P: dict, out_dir, meta: str = "") -> dict:
    from pathlib import Path
    out = {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for w in ("fp32", "fp16", "int8"):
        path = Path(out_dir) / f"sama_net_{w}.onnx"
        path.write_bytes(build_model(P, weights=w, meta=meta))
        out[w] = str(path)
    return out


def run_onnx(path, X, cond):
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return s.run(["y"], {"x": X.astype(np.float32), "cond": cond.astype(np.float32)})[0]
