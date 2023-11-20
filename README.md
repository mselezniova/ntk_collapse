# Neural (Tangent Kernel) Collapse

Source code for the paper ["Neural (Tangent Kernel) Collapse"](https://arxiv.org/abs/2305.16427) by Mariia Seleznova, Dana Weitzner, Raja Giryes, Gitta Kutyniok, Hung-Hsu Chou, *NeurIPS 2023*.

## Requirements

To install requirements in a new conda envoronment:

```setup
conda create --name <env> --file requirements.txt
```
GPU support requires external installations of CUDA 11.4+ and CuDNN 8.2+.

## Code

We include the following Python scripts: 
- ResNet20: ```resnet_ntk_collapse.py```
- VGG11/16: ```vgg_ntk_collapse.py```
- DenseNet40: ```densenet_ntk_collapse.py```

Each script trains a model, computes the metrics reported in the paper during training, and saves them in a pickle file. See implementation details in Appendix C. To run the scripts, use this command:

```train
python <architecture>_ntk_collapse.py --dataset <dataset_name> --init <initialization_name> --lr <learning_rate> --path <folder> --filename <file_name>
```
Arguments:
- ```--dataset```: dataset name as in tfds (values: mnist, FashionMNIST, cifar10)
- ```--init```: initialization setting (values: flax, normal_fan_out, torch)
- ```--lr```: learning rate initial value (float)
- ```--path```: path to a folder to save the experiment
- ```--filename```: file name to save the experiment

Example:
```
python resnet_ntk_collapse.py --dataset mnist  --init flax --lr 0.01  --path /path/to/folder/ --filename resnet20_mnist_lr001
```



