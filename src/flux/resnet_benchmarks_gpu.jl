include("resnetmodels.jl")
include("benchmarking_gpu.jl")

using Flux, CuArrays

batch_size = 8

run_benchmarks([ResNet18, ResNet34, ResNet50, ResNet101, ResNet152], batch_size)
