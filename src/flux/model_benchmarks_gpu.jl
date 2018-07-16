include("vggmodels.jl")
include("resnetmodels.jl")
include("benchmarking_gpu.jl")

using Flux, CuArrays

batch_size = 8

run_benchmarks([vgg16, vgg16_bn, vgg19, vgg19_bn, ResNet18, ResNet34, ResNet50], batch_size)

batch_size = 4

run_benchmarks([ResNet101, ResNet152], batch_size)
