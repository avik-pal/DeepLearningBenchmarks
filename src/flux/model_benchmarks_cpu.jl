include("vggmodels.jl")
include("resnetmodels.jl")
include("benchmarking_cpu.jl")

using Flux

batch_size = 8

run_benchmarks([vgg16, vgg16_bn, vgg19, vgg19_bn, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152], batch_size)
