import logging
import time

import coloredlogs
import torch
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    vgg16,
    vgg16_bn,
    vgg19,
    vgg19_bn,
)

NITERS = 51
BATCH_SIZE = 8
GPU_BENCHMARK = True
BENCHMARKS = dict()


def run_benchmark(m):
    logging.info("Running benchmark for model - {}".format(m))

    BENCHMARKS[m] = dict()

    forward_pass = []
    backward_pass = []
    total_time = []

    model = eval(m)()
    input_tensor = torch.randn(BATCH_SIZE, 3, 224, 224)
    grad_tensor = torch.ones(BATCH_SIZE, 1000)

    if GPU_BENCHMARK:
        model = model.cuda()
        input_tensor = input_tensor.cuda()
        grad_tensor = grad_tensor.cuda()

    for i in range(NITERS):
        start_time_forward = time.time()
        result = model(input_tensor)
        if GPU_BENCHMARK:
            torch.cuda.synchronize()
        end_time_forward = time.time()

        start_time_backward = time.time()
        result.backward(grad_tensor)
        if GPU_BENCHMARK:
            torch.cuda.synchronize()
        end_time_backward = time.time()

        if i == 1:
            continue

        forward_pass.append(end_time_forward - start_time_forward)
        backward_pass.append(end_time_backward - start_time_backward)
        total_time.append(forward_pass[-1] + backward_pass[-1])

    BENCHMARKS[m]["forward"] = min(forward_pass)
    BENCHMARKS[m]["backward"] = min(backward_pass)
    BENCHMARKS[m]["total"] = min(total_time)

    logging.info("Forward Pass Time = {}".format(BENCHMARKS[m]["forward"]))
    logging.info("Backward Pass Time = {}".format(BENCHMARKS[m]["backward"]))
    logging.info("Total Time = {}".format(BENCHMARKS[m]["total"]))


if __name__ == "__main__":
    coloredlogs.install()
    logging.getLogger().setLevel(logging.INFO)

    models = [
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    ]

    for model in models:
        run_benchmark(model)

    if GPU_BENCHMARK:
        filename = "pytorch_gpu_results.md"
    else:
        filename = "pytorch_cpu_results.md"

    file = open(filename, "w")

    file.write("# PYTORCH BENCHMARKS\n\n")

    file.write("| Model | Forward Pass | Backward Pass | Total Time |\n")

    for m in models:
        file.write(
            "| {} | {} | {} | {} |\n".format(
                m,
                BENCHMARKS[m]["forward"],
                BENCHMARKS[m]["backward"],
                BENCHMARKS[m]["total"],
            )
        )

    file.close()
