using BenchmarkTools

function run_benchmarks(arr, batch_size)
    for m in arr
        println("*"^10,m,"*"^10)
        model = m()
        input_arr = rand(224, 224, 3, batch_size)
        grad_arr = ones(1000, batch_size)

        println("Forward Pass Time :")
        # @btime result = $model($input_arr)
        t = zeros(10)
        for i in 1:10
            t[i] = @elapsed model(input_arr)
        end
        println(minimum(t))

        println("Total Time :")
        # @btime Flux.back!($model($input_arr), $grad_arr)
        for i in 1:10
            t[i] = @elapsed Flux.back!(model(input_arr), grad_arr)
        end
        println(minimum(t))
    end

    for m in arr
        println("*"^10,m,"*"^10)
        model = m()
        Flux.testmode!(model)
        input_arr = rand(224, 224, 3, batch_size)

        println("Inference Time :")
        # @btime result = $model($input_arr)
        t = zeros(10)
        for i in 1:10
            t[i] = @elapsed model(input_arr)
        end
        println(minimum(t))
    end
end
