# Python Backend
Idea is to simple "serialize" the full onnx graph in an anonymouse zig struct.
I will just adopt their entire naming convention to save myself a headache.


# Onnx Operators
## Implemented
* `Abs`
* `Acos`
* `Acosh`
* `Add`
* `Asin`
* `Asinh`
* `Atan`
* `Atanh`
* `Ceil`
* `Mul`

## TODO
* `AveragePool`
* `BatchNormalization`
* `Cast`
* `Celu`
* `Clip`
* `Concat`
* `Constant`
* `Conv`
* `ConvTranspose`
* `Cos`
* `Cosh`
* `CumSum`
* `Div`
* `Dropout`
* `Einsum`
* `Elu`
* `Erf`
* `Exp`
* `Floor`
* `Flatten`
* `GRU`
* `Gather`
* `GatherElements`
* `GatherND`
* `Gelu`
* `Gemm`
* `GlobalAveragePool`
* `GlobalMaxPool`
* `HardSigmoid`
* `HardSwish`
* `Hardmax`
* `Identity`
* `LSTM`
* `LayerNormalization`
* `LeakyRelu`
* `Log`
* `LogSoftmax`
* `MatMul`
* `Max`
* `MaxPool`
* `PRelu`
* `Pad`
* `Pow`
* `RNN`
* `Reciprocal`
* `Relu`
* `Reshape`
* `Resize`
* `Round`
* `Scan`
* `Scatter`
* `ScatterElements`
* `ScatterND`
* `Selu`
* `SoftmaxCrossEntropyLoss`
* `Tan`
* `Tanh`
* `Sqrt`
* `Squeeze`
* `Sub`
* `Sum`
* `ThresholdedRelu`
* `Transpose`
* `Trilu`
* `Unsqueeze`
* `Upsample`

## Skipped: Booleans and Integers are Future Work
* `And`: 
* `BitShift`
* `BitwiseAnd`
* `BitwiseNot`
* `BitwiseOr`
* `BitwiseXor`
* `Compress`
* `ConvInteger`
* `Equal`
* `Greater`
* `GreaterOrEqual`
* `InstanceNormalization`
* `If`
* `IsInf`
* `IsNaN`
* `Less`
* `LessOrEqual`
* `MatMulInteger`
* `Mean`
* `Min`
* `Mish`
* `Mod`
* `Neg`
* `NonZero`
* `Not`
* `Or`
* `Xor`
* `Where`

## Skipped: Deemed Niche, Future work
* `AffineGrid`:
* `ArgMax`
* `ArgMin`: 
* `Bernoulli`
* `BlackmanWindow`
* `CastLike`
* `CenterCropPad`
* `Col2Im`
* `ConcatFromSequence`
* `ConstantOfShape`
* `DFT`
* `DeformConv`
* `DepthToSpace`
* `DequantizeLinear`
* `Det`
* `DynamicQuantizeLinear`
* `Expand`
* `EyeLike`
* `GlobalLpPool`
* `GridSample`
* `GroupNormalization`
* `HammingWindow`
* `HannWindow`
* `ImageDecoder`
* `LRN`
* `Loop`
* `LpNormalization`
* `LpPool`
* `MaxRoiPool`
* `MaxUnpool`
* `MeanVarianceNormalization`
* `MelWeightMatrix`
* `Multinomial`
* `NegativeLogLikelihoodLoss`
* `NonMaxSuppression`
* `OneHot`
* `Optional`
* `OptionalGetElement`
* `OptionalHasElement`
* `QLinearConv`
* `QLinearMatMul`
* `QuantizeLinear`
* `RandomNormal`
* `RandomNormalLike`
* `RandomUniform`
* `RandomUniformLike`
* `Range`
* `ReduceL1`
* `ReduceL2`
* `ReduceLogSum`
* `ReduceLogSumExp`
* `ReduceMax`
* `ReduceMean`
* `ReduceMin`
* `ReduceProd`
* `ReduceSum`
* `ReduceSumSquare`
* `RegexFullMatch`
* `ReverseSequence`
* `RoiAlign`
* `STFT`
* `SequenceAt`
* `SequenceConstruct`
* `SequenceEmpty`
* `SequenceErase`
* `SequenceInsert`
* `SequenceLength`
* `SequenceMap`
* `Shape`
* `Shrink`
* `Sigmoid`
* `Sign`
* `Sin`
* `Sinh`
* `Size`
* `Slice`
* `Softmax`
* `Softplus`
* `Softsign`
* `SpaceToDepth`
* `Split`
* `SplitToSequence`
* `StringConcat`
* `StringNormalizer`
* `StringSplit`
* `TfIdfVectorizer`
* `Tile`
* `TopK`
* `Unique`
