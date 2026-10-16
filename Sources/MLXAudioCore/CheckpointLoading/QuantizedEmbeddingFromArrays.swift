import Foundation
import MLX
import MLXNN

/// A quantized embedding built from arrays the caller provides, the way
/// `QuantizedLinear.init(weight:bias:scales:biases:groupSize:bits:mode:)` allows
/// for linear layers.
///
/// mlx-swift only gains the equivalent `QuantizedEmbedding` initializer in
/// 0.32.2; until the package moves there, this mirrors `QuantizedEmbedding`'s
/// forward pass so a quantized embedding can start from placeholders too.
/// Delete it after the bump and use `QuantizedEmbedding` directly.
public final class QuantizedEmbeddingFromArrays: Embedding, Quantized {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    public let scales: MLXArray
    public let biases: MLXArray?

    public init(
        weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.scales = scales
        self.biases = biases
        super.init(weight: weight)
        freeze()
    }

    public override var shape: (Int, Int) {
        let (embeddingCount, packedDimensions) = super.shape
        return (embeddingCount, packedDimensions * 32 / bits)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let inputShape = x.shape
        let indices = x.flattened()
        let output = dequantized(
            weight[indices], scales: scales[indices], biases: biases.map { $0[indices] },
            groupSize: groupSize, bits: bits, mode: mode)
        return output.reshaped(inputShape + [-1])
    }

    public override func asLinear(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits, mode: mode)
    }
}
