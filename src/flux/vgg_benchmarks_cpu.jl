include("vggmodels.jl")
include("benchmarking_cpu.jl")

using Flux, BenchmarkTools

batch_size = 8

run_benchmarks([vgg16, vgg16_bn, vgg19, vgg19_bn])
