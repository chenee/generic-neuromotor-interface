from __future__ import annotations

import json
from pathlib import Path

REPORT_PATH = Path("/Users/chenee/Documents/Dev/算法/myEMG/mini/mini_rnn_float32.tflite.report.json")


def main() -> None:
    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    inputs = data.get("inputs", [])
    outputs = data.get("outputs", [])
    operator_counts = data.get("operator_counts", {})
    operators = data.get("operators", [])
    buffers = data.get("buffers", [])

    # Top operator types
    top_ops = sorted(operator_counts.items(), key=lambda x: (-x[1], x[0]))

    # Find Conv2D and FullyConnected operators and infer weight shapes
    conv_ops = []
    fc_ops = []
    for op in operators:
        name = op.get("op", {}).get("builtin")
        ins = op.get("inputs", [])
        outs = op.get("outputs", [])
        if name == "CONV_2D":
            # Expect inputs: input, filter, bias
            conv_ops.append({
                "index": op.get("index"),
                "input": ins[0] if len(ins) > 0 else None,
                "filter": ins[1] if len(ins) > 1 else None,
                "bias": ins[2] if len(ins) > 2 else None,
                "output": outs[0] if len(outs) > 0 else None,
            })
        elif name == "FULLY_CONNECTED":
            # Expect inputs: input, weights, bias
            fc_ops.append({
                "index": op.get("index"),
                "input": ins[0] if len(ins) > 0 else None,
                "weights": ins[1] if len(ins) > 1 else None,
                "bias": ins[2] if len(ins) > 2 else None,
                "output": outs[0] if len(outs) > 0 else None,
            })

    # Top buffers by size
    top_buffers = sorted(buffers, key=lambda b: b.get("size_bytes", 0), reverse=True)[:10]

    summary = {
        "model_version": data.get("model_version"),
        "subgraphs": data.get("subgraphs"),
        "operator_total": data.get("operator_total"),
        "tensor_total": data.get("tensor_total"),
        "inputs": inputs,
        "outputs": outputs,
        "top_operator_types": top_ops[:12],
        "conv2d_ops": conv_ops,
        "fully_connected_ops": fc_ops,
        "top_buffers": top_buffers,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
