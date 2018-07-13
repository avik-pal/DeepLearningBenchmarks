using BenchmarkTools

function run_benchmarks(arr, batch_size)
    for m in arr
        println("*"^10,m,"*"^10)
        model = m()
        input_arr = rand(224, 224, 3, batch_size)
        grad_arr = ones(1000, batch_size)

        println("Forward Pass Time :")
        @btime result = $model($input_arr)

        result = model(input_arr)

        println("Backward Pass Time :")
        @btime Flux.back!($result, $grad_arr)
    end

    for m in arr
        println("*"^10,m,"*"^10)
        model = m()
        Flux.testmode!(model)
        input_arr = rand(224, 224, 3, batch_size)

        println("Inference Time :")
        @btime result = $model($input_arr)
    end
end
