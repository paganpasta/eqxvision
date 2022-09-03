# Image Classification

## Imagenet-1K 


!!! Warning "Note"

    -   For `Vgg` and `Googlenet`, there's a big gap in performance of 
        pre-trained networks. The difference arises after the `adaptive-pooling`,
        which implies the networks can still be used as feature extractors 
        (see results [here](./getting_started/Transfer_Learning.ipynb)).
    
    -    For `RegNets`, the pretrained weights correspond to torchvision's `IMAGENET1K_V2`.
    
    -   `Swin_v2` pretrained is not supported. 
    
    -   `ViT` only supports `DINO` pretrained weights.


| Method             | Torchvision | Eqxvision  |
|--------------------|-------------|------------|
| Alexnet            | 56.518      | 56.522     |
| Convnext_tiny      | 82.132      | 82.120     |
| Densenet121        | 74.432      | 74.434     |
| Efficientnet_b0    | 77.686      | 77.684     |
| Efficientnet_v2_s  | 81.314      | 81.312     |
| Googlenet          | 69.774      | 62.462     |
| Mobilenet_v2       | 71.878      | 71.856     |
| Mobilenet_v3_small | 67.674      | 67.668     |
| Regnet_X_400MF     | 74.864      | 74.874     |
| Regnet_Y_400MF     | 75.806      | 75.800     |
| Resnet18           | 69.766      | 69.758     |
| Shufflenet_v2_x0_5 | 60.550      | 60.552     |
| Squeezenet_1_0     | 58.102      | 58.092     |
| Squeezenet_1_1     | 58.178      | 58.178     |
| Swin_T             | 81.474      | 81.172     |
| Swin_v2_T          | 82.072      | :no_entry: |
| Vgg_11             | 69.024      | 27.190     |
| Vgg_11_bn          | 70.376      | 57.726     |
