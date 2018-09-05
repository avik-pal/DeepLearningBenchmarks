## Input Dimensions
1. Batch Size = 8, Image = 3 x 224 x 224 (IF NOTHING SPECIFIED / CPU USED)
2. Batch Size = 4, Image = 3 x 224 x 224
    * Resnet 101
    * Resnet 152

## GPU USED --- Titan 1080Ti 12 GB
|Model|Framework|Forward Pass|Backward Pass|Total Time|Inference|
|:---:|:---:|:---:|:---:|:---:|:---:|
|VGG16|Pytorch 0.4|0.0245 s|0.0606 s|0.0852 s|0.0234 s|
||Flux|0.0802 s|0.5375 s|0.6177 s||
|VGG16 BN|Pytorch 0.4|0.0271 s|0.0672 s|0.0943 s|0.0273 s|
||Flux|0.0924 s|0.5582 s|0.6506 s||
|VGG19|Pytorch 0.4|0.0281 s|0.0741 s|0.1021 s|0.0280 s|
||Flux|0.0925 s|0.5375 s|0.6300 s||
|VGG19 BN|Pytorch 0.4|0.0321 s|0.0812 s|0.1134 s|0.0325 s|
||Flux|0.0975 s|0.5903 s|0.6878 s||
|Resnet18|Pytorch 0.4|0.0064 s|0.0125 s|0.0190 s|0.0050 s|
||Flux|0.0221 s|0.1306 s|0.1527 s|0.0219 s|
|Resnet34|Pytorch 0.4|0.0092 s|0.0216 s|0.0307 s|0.0092 s|
||Flux|0.0361 s|0.3000 s|0.3361 s|0.0357 s|
|Resnet50|Pytorch 0.4|0.0155 s|0.0351 s|0.0506 s|0.0152 s|
||Flux|0.4778 s|1.7582 s|2.206 s|0.6238 s|
|Resnet101|Pytorch 0.4|0.0297 s|0.0379 s|0.0676 s|0.0298 s|
||Flux|0.0720 s|0.6926 s|0.7646 s|0.0708 s|
|Resnet152|Pytorch 0.4|0.0431 s|0.05337 s|0.0965 s|0.0429 s|
||Flux|||||

## CPU USED --- Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz
|Model|Framework|Forward Pass|Backward Pass|Total Time|Inference|
|:---:|:---:|:---:|:---:|:---:|:---:|
|VGG16|Pytorch 0.4|6.6024 s|9.4336 s|16.036 s|6.4216 s|
||Flux|10.458 s|10.245 s|20.703 s|10.111 s|
|VGG16 BN|Pytorch 0.4|7.0793 s|9.0536 s|16.132 s|6.7909 s|
||Flux|29.633 s|18.649 s|49.282 s|24.047 s|
|VGG19|Pytorch 0.4|8.3075 s|10.899 s|19.2065 s|8.0593 s|
||Flux|12.226 s|12.457 s|24.683 s|12.029 s|
|VGG19 BN|Pytorch 0.4|8.7794 s|12.739 s|21.519 s|8.4044 s|
||Flux|28.518 s|21.464 s|49.982 s|22.649 s|
|Resnet18|Pytorch 0.4|||||
||Flux|||||
|Resnet34|Pytorch 0.4|||||
||Flux|||||
|Resnet50|Pytorch 0.4|||||
||Flux|||||
|Resnet101|Pytorch 0.4|||||
||Flux|||||
|Resnet152|Pytorch 0.4|||||
||Flux|||||

## Algorithm Benchmarks

### Layer Descriptions
1. Conv3x3/1 = Conv2d, 3x3 Kernel, 1x1 Padding, 1x1 Stride
2. Conv5x5/1 = Conv2d, 5x5 Kernel, 2x2 Padding, 1x1 Stride
3. Conv3x3/2 = Conv2d, 3x3 Kernel, 1x1 Padding, 2x2 Stride
4. Conv5x5/2 = Conv2d, 5x5 Kernel, 2x2 Padding, 2x2 Stride
5. Dense = 1024 => 512
6. BatchNorm = BatchNorm2d

## GPU USED --- Titan 1080Ti 12 GB
|Layer|Framework|Forward Pass|Backward Pass|Total Time|
|:---:|:---:|:---:|:---:|:---:|
|Conv3x3/1|Pytorch 0.4|0.2312 ms|0.5359 ms|0.7736 ms|
||Flux|0.405 ms|1.085 ms|1.490 ms|
|Conv5x5/1|Pytorch 0.4|0.2667 ms|0.5345 ms|0.8299 ms|
||Flux|0.447 ms|1.210 ms|1.657 ms|
|Conv3x3/2|Pytorch 0.4|0.1170 ms|0.2203 ms|0.3376 ms|
||Flux|0.145 ms|0.357 ms|0.502 ms|
|Conv5x5/2|Pytorch 0.4|0.1233 ms|0.2162 ms|0.3407 ms|
||Flux|0.154 ms|0.392 ms|0.546 ms|
|Dense|Pytorch 0.4|0.0887 ms|0.1523 ms|0.2411 ms|
||Flux|0.1179 ms|0.0875 ms|0.2054 ms|
|BatchNorm|Pytorch 0.4|0.1096 ms|0.1999 ms|0.3095 ms|
||Flux|0.2485 ms|0.2293 ms|0.4778 ms|

## CPU USED --- Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz
|Layer|Framework|Forward Pass|Backward Pass|Total Time|
|:---:|:---:|:---:|:---:|:---:|
|Conv3x3/1|Pytorch 0.4||||
||Flux||||
|Conv5x5/1|Pytorch 0.4||||
||Flux||||
|Conv3x3/2|Pytorch 0.4||||
||Flux||||
|Conv5x5/2|Pytorch 0.4||||
||Flux||||
|Dense|Pytorch 0.4||||
||Flux||||
|BatchNorm|Pytorch 0.4||||
||Flux||||

# NOTE

To reproduce the benchmarks checkout both `Flux` and `CuArrays` __master__.
