# ImageNet Dataset

* Train Set: Untar `ILSVRC2012_img_train.tar` somewhere and link it to the folder with name `train` (you can find a link file called train)
* Val Set: Untar `ILSVRC2012_img_val.tar` and link it to the folder with name `val`
  * Copy `valprep.sh` script to val folder and run it. It will put images to their own folder

# Mnist Dataset
Mnist dataset is imported from torchvision.datasets

# Train
1. To do the distillation experiment on the ImageNet Dataset

    `python main_imagenet.py ./`
2. To do the distillation experiment on the Mnist Dataset

    `python main_mnist.py ./`

# Reference
1. Code

`main_imagenet.py`: https://github.com/pytorch/examples/blob/master/imagenet/main.py
2. Paper

https://arxiv.org/abs/1503.02531
