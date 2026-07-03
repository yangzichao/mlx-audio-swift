import MLX

public struct STTGenerateParameters: Sendable {
    public let maxTokens: Int
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public let verbose: Bool
    public let language: String
    public let chunkDuration: Float
    public let minChunkDuration: Float
    /// Optional text context prepended to the decode prompt (e.g. preceding
    /// transcript, domain vocabulary). Models without a context channel
    /// ignore it. GLM-ASR renders it as a `<|system|>` turn ahead of the
    /// audio user turn.
    public let context: String?

    public init(
        maxTokens: Int = 8192,
        temperature: Float = 0.0,
        topP: Float = 0.95,
        topK: Int = 0,
        verbose: Bool = false,
        language: String = "English",
        chunkDuration: Float = 1200.0,
        minChunkDuration: Float = 1.0,
        context: String? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.verbose = verbose
        self.language = language
        self.chunkDuration = chunkDuration
        self.minChunkDuration = minChunkDuration
        self.context = context
    }
}

public protocol STTGenerationModel: AnyObject {
    var defaultGenerationParameters: STTGenerateParameters { get }

    func generate(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> STTOutput

    func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error>
}

public extension STTGenerationModel {
    func generate(
        audio: MLXArray,
        generationParameters: STTGenerateParameters? = nil
    ) -> STTOutput {
        generate(audio: audio, generationParameters: generationParameters ?? defaultGenerationParameters)
    }

    func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters? = nil
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        generateStream(audio: audio, generationParameters: generationParameters ?? defaultGenerationParameters)
    }
}
