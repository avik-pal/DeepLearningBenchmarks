## Input Dimensions : Batch Size = 8, Image = 3 x 224 x 224

## GPU USED --- Tesla V100 16 GB
|Model|Framework|Forward Pass|Backward Pass|Total Time|Inference|
|:---:|:---:|:---:|:---:|:---:|:---:|
|VGG16|Pytorch 0.4|0.0259 s|0.0554 s|0.0814 s|0.0245 s|
||Flux|0.0479 s|0.6293 s|0.6772 s||
|VGG16 BN|Pytorch 0.4|0.0262 s|0.0579 s|0.0842 s|0.0262 s|
||Flux|0.0560 s|0.6376 s|0.6936 s||
|VGG19|Pytorch 0.4|0.0329 s|0.0706 s|0.1035 s|0.0329 s|
||Flux|0.0565 s|0.6611 s|0.7176 s||
|VGG19 BN|Pytorch 0.4|0.0344 s|0.0701 s|0.1046 s|0.0329 s|
||Flux|0.0585 s|0.6659 s|0.7284 s||

## GPU USED --- Tesla P100 16 GB
|Model|Framework|Forward Pass|Backward Pass|Total Time|Inference|
|:---:|:---:|:---:|:---:|:---:|:---:|
|VGG16|Pytorch 0.4|0.0257 s|0.0699 s|0.0957 s|0.0246 s|
||Flux|0.0876 s|0.7596 s|0.8472 s||
|VGG16 BN|Pytorch 0.4|0.0274 s|0.0741 s|0.1015 s|0.0275 s|
||Flux|0.1092 s|0.7771 s|0.8863 s||
|VGG19|Pytorch 0.4|0.0299 s|0.0856 s|0.1155 s|0.0299 s|
||Flux|0.1012 s|0.8060 s|0.9072 s||
|VGG19 BN|Pytorch 0.4|0.0329 s|0.0902 s|0.1231 s|0.0329 s|
||Flux|0.1051 s|0.8116 s|0.9167 s||

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
