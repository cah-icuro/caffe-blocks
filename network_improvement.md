# Caffe L-Blocks

## Improving the Network

Goal: improve the network to beat naive classifier.

### Starting Point

Our images are binary images of an L-shaped tetris block (or no block) with random amounts of randomly placed single-pixel black/white noise.

The [naive classifier](https://github.com/cah-icuro/caffe-blocks/blob/master/naive_classify.py) simply looks for the presence of an L-shape in the image and classifies accordingly.  This method achieves **91.4%** accuracy, so this is our point of reference to beat.

We start with the [single-convolution net](https://github.com/cah-icuro/caffe-blocks/blob/master/nets/_train_class_net_v2.prototxt) shown below, which achieved an accuracy of **86.5%** (after 30K iterations of Adam solver).  The pooling layer is just a placeholder, its filter size is 1x1.
![net](/imgs/net_v2.png)

### Manual Filter Modification

In light of this large accuracy gap between what seems like a more powerful model and a naive model, I would like to see if this net can actually model the naive method by manually setting the filters and then retraining with the top convolutional layer frozen.
