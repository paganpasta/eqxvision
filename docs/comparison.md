# Image Classification

## Imagenet-1K 


!!! Warning "Note"

    -   For `Vgg` and `Googlenet`, there's a big gap in performance of 
        pre-trained networks. The difference arises after the `adaptive-pooling`,
        which implies the networks can still be used as feature extractors.
        `[PENDING]` Evaluation after fine-tuning the networks to confirm 
        the quality of extracted features.

    -   As `Mobilenet-v3` uses `adaptive-pooling` in predominantly every block,
        the pretrained model is as good as untrained one.


| Method            | Torchvision | Eqxvision  |
|-------------------|-------------|------------|
| Alexnet           | 56.518      | 56.522     |
| Densenet121       | 74.432      | 74.434     |
| Googlenet         | 69.774      | 61.046     |
| Mobilenet_v2      | 71.878      | 71.856     |
| Mobilenet_v3_small | 67.674      | :no_entry: |
| Resnet18          | 69.766      | 69.758     |
| Shufflenet_v2_x0_5 | 60.550      | 60.552     |
| Squeezenet_1_0    | 58.102      | 57.052     |
| Squeezenet_1_1    | 58.178      | 58.178     |
| Vgg-11            | 69.024      | 27.190     |

