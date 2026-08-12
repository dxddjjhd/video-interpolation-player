"""生成合成 ONNX 模型，用于验证 RifeInterpolator 适配层。

生成两种变体（与真实 RIFE 导出的输入结构一致）：
- concat 模式: 单输入 [1,6,H,W] → 输出 [1,3,H,W]（两帧取平均）
- pair_ts 模式: img0/img1/timestep 三输入 → 输出 [1,3,H,W]

输出并非真实插帧（均值），仅用于测试模型加载/IO 适配/缩放归一化。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tests/data"


def _build_concat(path: Path) -> None:
    """输入 (1,6,H,W) → 拆成两半求平均 → 输出 (1,3,H,W)。"""
    # x 按通道拆: (1,6,H,W) -> Split axis=1, halves (1,3,H,W)+(1,3,H,W) -> Add -> Mul(0.5)
    x = helper.make_tensor_value_info("img", TensorProto.FLOAT, [1, 6, "H", "W"])
    y = helper.make_tensor_value_info("img_out", TensorProto.FLOAT, [1, 3, "H", "W"])

    split = helper.make_node("Split", ["img"], ["a", "b"], axis=1)
    add = helper.make_node("Add", ["a", "b"], ["s"])
    half = helper.make_node(
        "Constant", [], ["half"],
        value=helper.make_tensor("half", TensorProto.FLOAT, [], [0.5]))
    mul = helper.make_node("Mul", ["s", "half"], ["img_out"])

    graph = helper.make_graph(
        [split, add, half, mul], "mean_interp",
        [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def _build_pair_ts(path: Path) -> None:
    """三输入 img0/img1/timestep → 线性混合 (1-t)*img0 + t*img1。"""
    i0 = helper.make_tensor_value_info("img0", TensorProto.FLOAT, [1, 3, "H", "W"])
    i1 = helper.make_tensor_value_info("img1", TensorProto.FLOAT, [1, 3, "H", "W"])
    ts = helper.make_tensor_value_info("timestep", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("img_out", TensorProto.FLOAT, [1, 3, "H", "W"])

    one = helper.make_node("Constant", [], ["one"],
                           value=helper.make_tensor("one", TensorProto.FLOAT, [], [1.0]))
    inv = helper.make_node("Sub", ["one", "timestep"], ["inv_ts"])
    wa = helper.make_node("Mul", ["img0", "inv_ts"], ["wa"])
    wb = helper.make_node("Mul", ["img1", "timestep"], ["wb"])
    out = helper.make_node("Add", ["wa", "wb"], ["img_out"])

    graph = helper.make_graph(
        [one, inv, wa, wb, out], "blend_interp",
        [i0, i1, ts], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _build_concat(OUT_DIR / "synthetic_concat.onnx")
    _build_pair_ts(OUT_DIR / "synthetic_pair_ts.onnx")
    print("合成模型已生成:",
          OUT_DIR / "synthetic_concat.onnx",
          OUT_DIR / "synthetic_pair_ts.onnx")


if __name__ == "__main__":
    main()
