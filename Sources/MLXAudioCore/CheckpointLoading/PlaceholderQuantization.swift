import Foundation
import MLX
import MLXNN

/// Replaces the layers selected by `filter` with quantized layers whose arrays
/// are lazy zero placeholders of the shapes the checkpoint fills in.
///
/// `quantize(model:filter:)` would instead run `MLX.quantized` on each random
/// init weight. The three outputs of that one lazy op are siblings that keep
/// each other alive, and `update(parameters:)` overwrites them in place, which
/// bypasses the sibling release in `mlx::core::array::~array`: the unevaluated
/// graph of every quantized layer outlived the model, on every load. The
/// checkpoint replaces all three arrays anyway, so nothing is lost by starting
/// from placeholders.
public func quantizeWithPlaceholderArrays(
    model: Module,
    filter: (String, Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)?
) {
    quantize(model: model, filter: filter, apply: placeholderQuantizedLayer(replacing:groupSize:bits:mode:))
}

private func placeholderQuantizedLayer(
    replacing layer: Module, groupSize: Int, bits: Int, mode: QuantizationMode
) -> Module? {
    if layer is Quantized {
        return nil
    }
    switch layer {
    case let linear as Linear:
        let (outputDimensions, inputDimensions) = linear.shape
        let arrays = PlaceholderQuantizedArrays(
            rows: outputDimensions, columns: inputDimensions, groupSize: groupSize, bits: bits, mode: mode)
        let quantized = QuantizedLinear(
            weight: arrays.weight, bias: linear.bias, scales: arrays.scales, biases: arrays.biases,
            groupSize: groupSize, bits: bits, mode: mode)
        quantized.freeze()
        return quantized
    case let embedding as Embedding:
        let (embeddingCount, dimensions) = embedding.shape
        let arrays = PlaceholderQuantizedArrays(
            rows: embeddingCount, columns: dimensions, groupSize: groupSize, bits: bits, mode: mode)
        return QuantizedEmbeddingFromArrays(
            weight: arrays.weight, scales: arrays.scales, biases: arrays.biases,
            groupSize: groupSize, bits: bits, mode: mode)
    default:
        // Anything else keeps the library's own quantization, leak included.
        return quantizeSingle(layer: layer, groupSize: groupSize, bits: bits, mode: mode)
    }
}

/// Zero arrays with the shapes `MLX.quantized` produces for a `[rows, columns]` matrix.
struct PlaceholderQuantizedArrays {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?

    init(rows: Int, columns: Int, groupSize: Int, bits: Int, mode: QuantizationMode) {
        let groupCount = columns / groupSize
        let isAffine = mode == .affine
        weight = MLXArray.zeros([rows, columns * bits / 32], dtype: .uint32)
        scales = MLXArray.zeros([rows, groupCount], dtype: isAffine ? .bfloat16 : .uint8)
        biases = isAffine ? MLXArray.zeros([rows, groupCount], dtype: .bfloat16) : nil
    }
}
