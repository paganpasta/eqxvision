# Image Classification

!!! Info "Note"
    Due to slight differences in the implementation of underlying operations,
    the resulting performance may vary.


## Imagenet-1K 

| Method         | Torchvision | Eqxvision |
|----------------|-------------|-----------|
| Alexnet        | 56.518      | 56.522    |
| SqueezeNet-1.0 | 58.102      | 57.052    |
| SqueezeNet-1.1 | 58.178      | 58.178    |
| Vgg-11         | 69.024      | 27.190    |
| Vgg-13         | 69.932      | 24.774    |
| Vgg-16         | 71.594      | 32.562    |
| Vgg-19         | 72.374      | 37.852    |


!!! Info "VGGs"
    The difference arises due to `Adaptive average pooling`, resulting
    in large deviation from `torchvision` classification performance. 
    The pre-trained models can still be used as feature extractors.
