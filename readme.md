# ImageNet Dataset

* Train Set: Untar `ILSVRC2012_img_train.tar` somewhere and link it to the folder with name `train` (you can find a link file called train)
* Val Set: Untar `ILSVRC2012_img_val.tar` and link it to the folder with name `val`
  * Copy `valprep.sh` script to val folder and run it. It will put images to their own folder

# Train

    python main.py ./

# Based on

https://github.com/pytorch/examples/blob/master/imagenet/main.py
