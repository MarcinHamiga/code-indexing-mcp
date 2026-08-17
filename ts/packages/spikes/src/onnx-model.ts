/**
 * A minimal, valid ONNX model built in-process.
 *
 * S0 asks for "one real call" through each native addon, and for
 * onnxruntime-node a real call means an actual inference -- loading the addon
 * proves dlopen worked, but not that ORT's native execution path runs under
 * Bun's N-API layer. Rather than commit a binary fixture or make the spike
 * depend on a 640 MB model download that belongs to S3, the graph is encoded
 * here directly: a single `Identity` node over a 1x4 float tensor.
 *
 * ONNX is protobuf, so this is a protobuf writer with exactly the four wire
 * shapes the ModelProto subset needs.
 */

function varint(value: number): number[] {
  const bytes: number[] = [];
  let remaining = value;
  do {
    let byte = remaining & 0x7f;
    remaining >>>= 7;
    if (remaining > 0) byte |= 0x80;
    bytes.push(byte);
  } while (remaining > 0);
  return bytes;
}

/** Wire type 0 -- a varint-valued field. */
function varintField(fieldNumber: number, value: number): number[] {
  return [...varint((fieldNumber << 3) | 0), ...varint(value)];
}

/** Wire type 2 -- a length-delimited field wrapping an already-encoded message. */
function nested(fieldNumber: number, payload: readonly number[]): number[] {
  return [...varint((fieldNumber << 3) | 2), ...varint(payload.length), ...payload];
}

function stringField(fieldNumber: number, value: string): number[] {
  return nested(fieldNumber, [...new TextEncoder().encode(value)]);
}

/** TypeProto for a float tensor of the given static shape. */
function floatTensorType(dims: readonly number[]): number[] {
  const shape = dims.flatMap((dim) => nested(1, varintField(1, dim)));
  // TypeProto.Tensor: elem_type = 1 (FLOAT), shape.
  const tensorType = [...varintField(1, 1), ...nested(2, shape)];
  // TypeProto.tensor_type
  return nested(1, tensorType);
}

function valueInfo(name: string, dims: readonly number[]): number[] {
  return [...stringField(1, name), ...nested(2, floatTensorType(dims))];
}

/**
 * Serialize the identity model. Returns the bytes an `InferenceSession` can be
 * created from directly, with no file ever touching disk.
 */
export function identityModel(dims: readonly number[] = [1, 4]): Uint8Array {
  // NodeProto: input "x", output "y", name, op_type "Identity".
  const node = [
    ...stringField(1, "x"),
    ...stringField(2, "y"),
    ...stringField(3, "identity"),
    ...stringField(4, "Identity"),
  ];

  // GraphProto: node, name, input, output.
  const graph = [
    ...nested(1, node),
    ...stringField(2, "spike"),
    ...nested(11, valueInfo("x", dims)),
    ...nested(12, valueInfo("y", dims)),
  ];

  // OperatorSetIdProto: default domain at opset 13, which IR version 8 admits.
  const opset = [...stringField(1, ""), ...varintField(2, 13)];

  // ModelProto: ir_version, producer_name, graph, opset_import.
  const model = [
    ...varintField(1, 8),
    ...stringField(2, "code-indexing-mcp-spike"),
    ...nested(7, graph),
    ...nested(8, opset),
  ];

  return Uint8Array.from(model);
}
