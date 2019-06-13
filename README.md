# COMPUTER VISION MODEL BENCHMARKS

## INPUT SPECIFICATIONS

1. All Inputs are **Float32**.
2. Batch Size is **8**.
3. Dimensions of the Image are **3 x 224 x 224**.

## RUNNING THE BENCHMARKS

### PYTORCH

1. Install the dependencies - Pytorch 1.1+ with/without GPU Support,
   torchvision and coloredlogs.
2. Change the `NITERS`, `BATCH_SIZE` and `GPU_BENCHMARK` (if needed).
3. Run with `python models.py`.

### FLUX

1. Move into the desired directory (divided as per Tracker/Zygote/CUDA)
   and do `instantiate` inside the Julia REPL.

## RESULTS
