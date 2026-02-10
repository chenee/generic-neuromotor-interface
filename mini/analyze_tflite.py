from pathlib import Path
import json
import tflite

MODEL_PATH = Path("/Users/chenee/Documents/Dev/算法/myEMG/mini/mini_rnn_float32.tflite")
REPORT_PATH = Path("/Users/chenee/Documents/Dev/算法/myEMG/mini/mini_rnn_float32.tflite.report.json")


def main() -> None:
    buf = MODEL_PATH.read_bytes()
    model = tflite.Model.GetRootAsModel(buf, 0)
    sg = model.Subgraphs(0)

    builtin_op_map = {
        v: k
        for k, v in tflite.BuiltinOperator.__dict__.items()
        if isinstance(v, int)
    }
    tensor_type_map = {
        v: k
        for k, v in tflite.TensorType.__dict__.items()
        if isinstance(v, int)
    }

    op_codes = []
    for i in range(model.OperatorCodesLength()):
        op = model.OperatorCodes(i)
        builtin = op.BuiltinCode()
        custom = op.CustomCode()
        custom = custom.decode("utf-8") if custom else None
        builtin_name = builtin_op_map.get(builtin, str(builtin))
        op_codes.append({"builtin": builtin_name, "custom": custom})

    def tensor_info(idx: int) -> dict:
        t = sg.Tensors(idx)
        name = t.Name().decode("utf-8") if t.Name() else ""
        shape = [t.Shape(i) for i in range(t.ShapeLength())]
        ttype = t.Type()
        ttype_name = tensor_type_map.get(ttype, str(ttype))
        q = t.Quantization()
        qscale = [q.Scale(i) for i in range(q.ScaleLength())] if q else []
        qzero = [q.ZeroPoint(i) for i in range(q.ZeroPointLength())] if q else []
        return {
            "idx": idx,
            "name": name,
            "shape": shape,
            "type": ttype_name,
            "quant_scale": qscale,
            "quant_zero": qzero,
        }

    inputs = [tensor_info(sg.Inputs(i)) for i in range(sg.InputsLength())]
    outputs = [tensor_info(sg.Outputs(i)) for i in range(sg.OutputsLength())]

    op_list = []
    operators = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        oc = op_codes[op.OpcodeIndex()]
        op_list.append(oc["builtin"])
        ins = [op.Inputs(j) for j in range(op.InputsLength())]
        outs = [op.Outputs(j) for j in range(op.OutputsLength())]
        operators.append({
            "index": i,
            "op": oc,
            "inputs": [tensor_info(idx) if idx >= 0 else {"idx": idx} for idx in ins],
            "outputs": [tensor_info(idx) if idx >= 0 else {"idx": idx} for idx in outs],
        })

    op_counts = {}
    for name in op_list:
        op_counts[name] = op_counts.get(name, 0) + 1

    buffers = []
    for i in range(model.BuffersLength()):
        b = model.Buffers(i)
        data = b.DataAsNumpy()
        if hasattr(data, "size"):
            size = int(data.size)
        else:
            size = int(b.DataLength()) if hasattr(b, "DataLength") else 0
        buffers.append({"index": i, "size_bytes": size})

    result = {
        "model_version": model.Version(),
        "subgraphs": model.SubgraphsLength(),
        "operator_codes": len(op_codes),
        "inputs": inputs,
        "outputs": outputs,
        "operator_counts": op_counts,
        "operator_total": sg.OperatorsLength(),
        "tensor_total": sg.TensorsLength(),
        "buffers": buffers,
        "operators": operators,
    }

    REPORT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(REPORT_PATH),
        "model_version": result["model_version"],
        "inputs": result["inputs"],
        "outputs": result["outputs"],
        "operator_total": result["operator_total"],
        "tensor_total": result["tensor_total"],
        "operator_counts": result["operator_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
