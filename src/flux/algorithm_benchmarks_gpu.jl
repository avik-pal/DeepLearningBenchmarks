using Flux, CuArrays, BenchmarkTools, CUDAnative

function run_benchmarks()
    println("Forward Pass :")
    @btime begin
        $layer($x)
        CUDAnative.synchronize()
    end

    println("Total Time :")
    @btime begin
        Flux.back!($layer($x), $grad)
        CUDAnative.synchronize()
    end
end

x = rand(224, 224, 3, 1) |> gpu
grad = ones(224, 224, 64, 1) |> gpu
layer = Conv((3,3), 3=>64, pad = (1, 1)) |> gpu
println("Benchmarks for Conv3x3/1")
run_benchmarks()

layer = Conv((5,5), 3=>64, pad = (2, 2)) |> gpu
println("Benchmarks for Conv5x5/1")
run_benchmarks()

layer = Conv((3,3), 3=>64, pad = (1, 1), stride = (2, 2)) |> gpu
grad = ones(112, 112, 64, 1) |> gpu
println("Benchmarks for Conv3x3/2")
run_benchmarks()

layer = Conv((5,5), 3=>64, pad = (2, 2), stride = (2, 2)) |> gpu
println("Benchmarks for Conv5x5/2")
run_benchmarks()

layer = MaxPool((3, 3), stride = (2, 2), pad = (1, 1)) |> gpu
grad = ones(112, 112, 3, 1) |> gpu
println("Benchmarks for Maxpool")
run_benchmarks()

layer = MeanPool((3, 3), stride = (2, 2), pad = (1, 1)) |> gpu
println("Benchmarks for Meanpool")
run_benchmarks()

grad = ones(224, 224, 3, 1) |> gpu
layer = BatchNorm(3) |> gpu
println("Benchmarks for BatchNorm")
run_benchmarks()

x = rand(1024, 1) |> gpu
grad = ones(512, 1) |> gpu
layer = Dense(1024, 512) |> gpu
println("Benchmarks for Dense")
run_benchmarks()
