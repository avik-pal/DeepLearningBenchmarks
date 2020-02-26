using Flux, CuArrays, Zygote, BenchmarkTools, BSON

const suite = BenchmarkGroup()

suite["forward pass"] = BenchmarkGroup()
suite["backward pass"] = BenchmarkGroup()
suite["total time"] = BenchmarkGroup()

function add!(suite, layer, layer_name, input)
    layer = layer |> gpu
    input = input |> gpu
    out, back = Zygote.pullback(layer, input)
    grad = ones(size(out)) |> gpu
    if CuArrays.functional()
        suite["forward pass"][layer_name] = @benchmarkable CuArrays.@sync $layer($input)
        suite["backward pass"][layer_name] = @benchmarkable CuArrays.@sync $back($grad)
        suite["total time"][layer_name] = @benchmarkable begin
            _, back = CuArrays.@sync Zygote.pullback($layer, $input)
            CuArrays.@sync back($grad)
        end
    else
        suite["forward pass"][layer_name] = @benchmarkable $layer($input)
        suite["backward pass"][layer_name] = @benchmarkable $back($grad)
        suite["total time"][layer_name] = @benchmarkable begin
            _, back = Zygote.pullback($layer, $input)
            back($grad)
        end
    end
    @info "Added Benchmark suite for " * layer_name
end


for batch_size in [1, 4, 16, 64, 256]
    add!(
        suite,
        Dense(1024, 512),
        "Dense (1024 --> 512) (Batch Size: $batch_size)",
        rand(MersenneTwister(1), 1024, batch_size)
    )

    input = rand(MersenneTwister(1), 224, 224, 3, batch_size)

    add!(
        suite,
        Conv((3, 3), 3=>64),
        "Conv2D (3x3/1) (Batch Size: $batch_size)",
        input
    )

    add!(
        suite,
        Conv((5, 5), 3=>64),
        "Conv2D (5x5/1) (Batch Size: $batch_size)",
        input
    )

    add!(
        suite,
        Conv((3, 3), 3=>64, stride = 2),
        "Conv2D (3x3/2) (Batch Size: $batch_size)",
        input
    )

    add!(
        suite,
        Conv((5, 5), 3=>64, stride = 2),
        "Conv2D (5x5/2) (Batch Size: $batch_size)",
        input
    )

    add!(
        suite,
        MaxPool((3, 3), stride = 2),
        "MaxPool2D (3x3/2) (Batch Size: $batch_size)",
        input
    )

    add!(
        suite,
        MeanPool((3, 3), stride = 2),
        "MeanPool2D (3x3/2) (Batch Size: $batch_size)",
        input
    )

    add!(
        suite,
        BatchNorm(3),
        "BatchNorm2D (Batch Size: $batch_size)",
        input
    )
end

if CuArrays.functional()
    paramspath = joinpath(dirname(@__FILE__), "params_gpu.json")
    results_file = joinpath(dirname(@__FILE__), "results_gpu.json")
else
    paramspath = joinpath(dirname(@__FILE__), "params_cpu.json")
    results_file = joinpath(dirname(@__FILE__), "results_cpu.json")
end

if isfile(paramspath)
    @info "Loading params from " * paramspath
    loadparams!(suite, BenchmarkTools.load(paramspath)[1], :evals);
else
    @info "Tuning the benchmark suite"
    tune!(suite)
    BenchmarkTools.save(paramspath, BenchmarkTools.params(suite));
end

results = run(suite, verbose=true)

BSON.@save results_file results
