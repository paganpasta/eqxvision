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
    There's a big gap in performance of pre-trained networks here.
    The features generated are `close` with a tolerance of `1e-1`. 
    `[PENDING]` Evaluation after fine-tuning the networks to assess the quality
    of features extracted for usability.
