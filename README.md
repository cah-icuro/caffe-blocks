# caffe-blocks

### No noise

<details>
  <summary>Training Network Definition</summary><p>
  
```
name: "CaffeNet"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 16
  }
  data_param {
    source: "/home/artemis/Programming/caffe/blocks_test/data/train_image_db"
    batch_size: 64
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 16
  }
  data_param {
    source: "/home/artemis/Programming/caffe/blocks_test/data/val_image_db"
    batch_size: 16
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 1
    stride: 1
  }
}
layer {
  name: "fc2"
  type: "InnerProduct"
  bottom: "pool1"
  top: "fc2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 256
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "fc2"
  top: "fc2"
}
layer {
  name: "fc3"
  type: "InnerProduct"
  bottom: "fc2"
  top: "fc3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 16
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "fc3"
  top: "fc3"
}
layer {
  name: "fc4"
  type: "InnerProduct"
  bottom: "fc3"
  top: "fc4"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "fc4"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "fc4"
  bottom: "label"
  top: "loss"
}
```
</p></details><br/>

<details>
  <summary>Depoloy Network Definition</summary><p>
  
```
name: "CaffeNet"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape: { dim: 1 dim: 1 dim: 16 dim: 16 } }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 1
    stride: 1
  }
}
layer {
  name: "fc2"
  type: "InnerProduct"
  bottom: "pool1"
  top: "fc2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 256
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "fc2"
  top: "fc2"
}
layer {
  name: "fc3"
  type: "InnerProduct"
  bottom: "fc2"
  top: "fc3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 16
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "fc3"
  top: "fc3"
}
layer {
  name: "fc4"
  type: "InnerProduct"
  bottom: "fc3"
  top: "fc4"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "loss"
  type: "Softmax"
  bottom: "fc4"
  top: "loss"
}
```
</p></details><br/>

Achieved 100% accuracy in <1K iterations, seemed to work perfectly on new images.

However, after running more tests, it seems likely that it's just looking to see if there's any white on the image at all (a trivial linear function can do this).

I decided to add noise and see if it can be retrained to be more robust.

### Noise in All Training Images

Same network as above.

Training data generated with significant amounts of noise in all images.

```python
def add_noise(img, ratio=0.1):
  h, w = img.shape[:2]
  num_noise_points = int(round(h * w * ratio))
  for i in range(num_noise_points):
    val = 255 * np.random.randint(2)
    x = np.random.randint(w)
    y = np.random.randint(h)
    img[y, x] = val
```

The results were interesting:

```
Image: test_images/0_1.png -- Class: 0 -- Confidence: 0.9999981
Image: test_images/1_1.png -- Class: 0 -- Confidence: 0.975722
Image: test_images/1_2.png -- Class: 0 -- Confidence: 0.99966455
Image: test_images/1_3.png -- Class: 0 -- Confidence: 0.92505187
Image: test_images/1_4.png -- Class: 0 -- Confidence: 0.9969131
Image: test_images/mirror_1_1.png -- Class: 1 -- Confidence: 0.6073349
Image: test_images/mirror_1_2.png -- Class: 0 -- Confidence: 0.7194996
Image: test_images/mirror_1_3.png -- Class: 0 -- Confidence: 0.69426894
Image: test_images/mirror_1_4.png -- Class: 0 -- Confidence: 0.9920913
Image: test_images/mirror_1_5.png -- Class: 0 -- Confidence: 0.98291194
Image: test_images/noisy_0_1.png -- Class: 0 -- Confidence: 0.8096211
Image: test_images/noisy_0_2.png -- Class: 0 -- Confidence: 0.9996117
Image: test_images/noisy_0_3.png -- Class: 0 -- Confidence: 0.99930406
Image: test_images/noisy_1_1.png -- Class: 1 -- Confidence: 0.8626109
Image: test_images/noisy_1_2.png -- Class: 1 -- Confidence: 0.5034988
Image: test_images/noisy_1_3.png -- Class: 1 -- Confidence: 0.8939095
Image: test_images/wild_1.png -- Class: 0 -- Confidence: 0.99999964
Image: test_images/wild_2.png -- Class: 0 -- Confidence: 0.9999987
Image: test_images/wild_3.png -- Class: 1 -- Confidence: 0.99984765
Image: test_images/wild_4.png -- Class: 0 -- Confidence: 0.6199456
Image: test_images/wild_5.png -- Class: 0 -- Confidence: 0.70783854

```

The images are names as `(prefix_)class_sequencenumber.png`.

The first 5 images are plain, noiseless images.  While the previous classifier actually identified these correctly, this one failed, labelling them all as zero (no object).  This makes me think it's still just doing some pixel-counting.

The next 5 `mirror_` images contain the object mirrored, also noiseless.  Seems the net did a little better on these for some reason, but might just be coincidence.

The next 6 `noisy_` images show decent results. It seems the net does decently well on noisy images (though the one 50.3% confidence doesn't count).  However, if the noise isn't super similar to the noise given in the training set, it fails, so we need to train it on data containing more varied noise.

The `wild_` images are just for experimentation, and don't have a clear correct label.

### Varying the Noise

We need to generate less regular noise:

```python
def add_noise(img, ratio=0.15):
  h, w = img.shape[:2]
  max_num_noise_points = int(round(h * w * ratio))
  num_noise_points = np.random.randint(max_num_noise_points + 1)
  for i in range(num_noise_points):
    val = 255 * np.random.randint(2)
    x = np.random.randint(w)
    y = np.random.randint(h)
    img[y, x] = val
```

Now we get the results:

```
Image: test_images/0_1.png -- Class: 0 -- Confidence: 1.0
Image: test_images/1_1.png -- Class: 1 -- Confidence: 0.9937826
Image: test_images/1_2.png -- Class: 1 -- Confidence: 0.99472344
Image: test_images/1_3.png -- Class: 1 -- Confidence: 0.99280965
Image: test_images/1_4.png -- Class: 1 -- Confidence: 0.99472344
Image: test_images/mirror_1_1.png -- Class: 1 -- Confidence: 0.99472344
Image: test_images/mirror_1_2.png -- Class: 1 -- Confidence: 0.99472344
Image: test_images/mirror_1_3.png -- Class: 1 -- Confidence: 0.99472344
Image: test_images/mirror_1_4.png -- Class: 1 -- Confidence: 0.99472344
Image: test_images/mirror_1_5.png -- Class: 1 -- Confidence: 0.9904129
Image: test_images/noisy_0_1.png -- Class: 0 -- Confidence: 0.99994826
Image: test_images/noisy_0_2.png -- Class: 0 -- Confidence: 1.0
Image: test_images/noisy_0_3.png -- Class: 0 -- Confidence: 0.9946622
Image: test_images/noisy_1_1.png -- Class: 0 -- Confidence: 0.937153
Image: test_images/noisy_1_2.png -- Class: 0 -- Confidence: 0.99947053
Image: test_images/noisy_1_3.png -- Class: 1 -- Confidence: 0.99472344
Image: test_images/wild_1.png -- Class: 0 -- Confidence: 0.9999994
Image: test_images/wild_2.png -- Class: 0 -- Confidence: 1.0
Image: test_images/wild_3.png -- Class: 0 -- Confidence: 0.73722804
Image: test_images/wild_4.png -- Class: 0 -- Confidence: 0.9999994
Image: test_images/wild_5.png -- Class: 0 -- Confidence: 0.99984515
```

The obvious question is what's different about `noisy_1_3` vs `noisy_1_1`, `noisy_1_2`?

Well, they're not really that different, especially 2 vs 3, so it must have overtrained.

Tried changing the `add_noise` function's `ratio=0.15` to `ratio=0.2`, got:

```
Image: test_images/noisy_0_1.png -- Class: 0 -- Confidence: 0.9999747
Image: test_images/noisy_0_2.png -- Class: 0 -- Confidence: 1.0
Image: test_images/noisy_0_3.png -- Class: 0 -- Confidence: 0.9999585
Image: test_images/noisy_1_1.png -- Class: 1 -- Confidence: 0.99061686
Image: test_images/noisy_1_2.png -- Class: 0 -- Confidence: 0.98299986
Image: test_images/noisy_1_3.png -- Class: 0 -- Confidence: 0.97431254
```

Within the noisy class 1 images 2 and 3 have 11 white noise pixels, whereas image 1 has 13.  Seems counting pixels is a problem, but it gets the noiseless ones right.

Now will try with 50K training iterations and lesser learning rate.

```
# Solver definition

net: "prototxt/train_class_net.prototxt"

# test batch size = 2^4
# test image count = 2^14
# --> test_iter = 2^10
test_iter: 1024
test_interval: 1000

# The base learning rate, momentum and the weight decay of the network.
base_lr: 0.003 # originally .01
momentum: 0.5 # originally 0.9
weight_decay: 0.0005
# The learning rate policy
lr_policy: "inv"
gamma: 0.0001
power: 0.75

# Display every 100 iterations
display: 100
# The maximum number of iterations
max_iter: 50000

# snapshot intermediate results
snapshot: 1000
snapshot_prefix: "snapshots"

solver_mode: GPU
```

Now for the noisy images we get the following (the rest of the classifications remain the same).

```
Image: test_images/noisy_0_1.png -- Class: 0 -- Confidence: 1.0
Image: test_images/noisy_0_2.png -- Class: 0 -- Confidence: 1.0
Image: test_images/noisy_0_3.png -- Class: 0 -- Confidence: 1.0
Image: test_images/noisy_1_1.png -- Class: 0 -- Confidence: 1.0
Image: test_images/noisy_1_2.png -- Class: 1 -- Confidence: 0.9999461
Image: test_images/noisy_1_3.png -- Class: 1 -- Confidence: 0.703021
```

We still have a partitioning of `noisy_1_1` vs `noisy_1_2, noisy_1_3`, but now it's reversed.  Maybe now it's learning something more complex, like if an image has more white pixels, then it's more likely that if those pixels happened to make a certain shape it was by coincidence...

The test accuracy of this final model was:

```
Test net output #0: accuracy = 0.821106
```

Let's see how this compares with the naive classifier: Look for an "L" in the image, if found return 1 else return 0.

Results:
```
artemis@artemis-pc:~/Programming/caffe/blocks_test$ python naive_classify.py 
Classifying images in data/val_imgs ...
Accuracy: 0.561340332031
Correct: 9197 / 16384
```

Wow, only 56% accuracy!  Did I do something wrong? TODO
