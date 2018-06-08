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

<details>
  <summary>Naive Classifier Code</summary><p>
  
```
from __future__ import print_function

import numpy as np
import cv2
import os
import re

IMAGE_WIDTH = 16
IMAGE_HEIGHT = 16

def list_files_of_type(dir, ext):
  filenames = sorted(f for f in os.listdir(dir) if f.endswith('.' + ext))
  return [os.path.join(dir, f) for f in filenames]

# Assumes image and filter are binary
def check_filter_match(img, filter):
  img = img > 0
  mask = filter > 0
  target_sum = np.sum(mask)
  img_h, img_w = img.shape
  filt_h, filt_w = filter.shape
  for i in range(img_h - filt_h + 1):
    for j in range(img_w - filt_w + 1):
      window = img[i:i+filt_h, j:j+filt_w]
      if (np.sum(mask & window) == target_sum):
        return True # Or (i, j) if we want coords
  return False

# Checks for the pattern existing in the image in any orientation
def check_pattern_in_image(img, pattern):
  for orientation in range(4):
    rotated_pattern = np.rot90(np.copy(pattern), orientation)
    if (check_filter_match(img, rotated_pattern)):
      return True
  return False

# Returns an image's class based on my filenaming conventions
# (First digit in a filename represents its class, wild images have no class)
def get_image_class(image_path):
  filename = image_path.split('/')[-1]
  if filename.startswith('wild'):
    return -1
  m = re.search('\d', filename)
  return int(m.group())


# PATTERN = np.array([[1, 1, 1], [1, 0, 0]], dtype='uint8')
PATTERN = np.array([[1, 1, 1], [0, 0, 1]], dtype='uint8')
IMAGE_DIR = os.path.join('data', 'val_imgs')
IMAGE_EXT = 'png' # extension without the '.'
image_files = list_files_of_type(IMAGE_DIR, IMAGE_EXT)
print('Classifying images in', IMAGE_DIR, '...')
correct_count = 0
total_count = 0
fp_count = 0
fn_count = 0
tp_count = 0
tn_count = 0
for image_path in image_files:
  image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  if (image.shape != (IMAGE_WIDTH, IMAGE_HEIGHT)):
    print('An image of incorrect shape was encountered, skipping...')
    print('(Shape must be', IMAGE_WIDTH, 'x', IMAGE_HEIGHT, ')')
    continue
  # Classify
  classification = int(check_pattern_in_image(image, PATTERN))
  correct_class = get_image_class(image_path)
  if (correct_class == -1): break
  correct_count += (classification == correct_class)
  # count true/false positives/negatives:
  if correct_class: # positive
    if classification: # true positive
      tp_count += 1
    else: # false negative
      fn_count += 1
  else: # negative
    if classification: # false positive
      fp_count += 1
    else: # true negative
      tn_count += 1
  # print('Image:', image_path, '-- Class:', classification, ', Correct:', correct_class)
  # Increment total count if we didn't skip this image
  total_count += 1

print('Accuracy:', correct_count*1.0/total_count)
print('Correct:', correct_count, '/', total_count)

fp_rate = (fp_count * 1.0) / (fp_count + tn_count)
fn_rate = (fn_count * 1.0) / (fn_count + tp_count)

print('False positive rate:', fp_rate)
print('False negative rate:', fn_rate)

print()
```
</p></details><br/>

Results:
```
Classifying images in data/val_imgs ...
Accuracy: 0.914489746094
Correct: 14983 / 16384
False positive rate: 0.00973117625593
False negative rate: 0.161827759402

```

We got 91% accuracy, with almost all of our errors being false negatives.  This is as expected, since if the noise ever obscured the L shape, we would classify it as a negative.  On the other hand, false positives can only occur if the noise happens to completely build the L shape.

This classifier was mathematically very simple, so it should be our benchmark for neural network performance.  For example, a neural network with a single convolutional layer having four convolutional filters that check for this shape in each of its four possible orientation, and then combines those activations with a single FC layer followed by softmax to classify the objects.  This should have the power to at least replicate the results from the above naive classifier.

## Improving the Network

Going back to look at the initial network, I immediately see that the architecture doesn't make sense.  The very first layer of the network was the following:

```
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
```

We're applying a set of 16 1x1 convolutional filters to our input image right off the bat.  This means we're dropping 15/16 of the pixels right from the very beginning, and still the neural network got over 80% accuracy - this is pretty impressive actually.

As a proof of concept, let's try to make the network make a little more sense given the objective (identifying noise images with the L shape), to see if we can beat the naive classifier benchmark.  We should start with some 3x3 convolutional filters, and zero-pad the image so we can detect the image on the corners.

## Solving

<details>
  <summary>Adam Solver</summary><p>
  
```
net: "prototxt/train_class_net.prototxt"

# training data count = 2^16
# test data count = 2^14
# test batch size = 2^4
# --> test_iter = 2^10, so we cover all test data per iteration
test_iter: 1024
test_interval: 1000

# Solver method parameters
type: "Adam"
momentum: 0.9
momentum2: 0.999
base_lr: 0.001
lr_policy: "fixed"

# Output info
display: 1000
max_iter: 30000

# snapshot intermediate results
snapshot: 5000
snapshot_prefix: "snapshots"

solver_mode: GPU
```
</p></details><br/>

<details>
  <summary>Accuracy: 0.86</summary><p>
  
```
Iteration: 0
				Accuracy: 0.497742
		Loss: 0.697259
		Loss: 0.696256
		Loss: 0.696256
Iteration: 1000
				Accuracy: 0.850281
		Loss: 0.332021
		Loss: 0.439961
		Loss: 0.439961
Iteration: 2000
				Accuracy: 0.853516
		Loss: 0.32909
		Loss: 0.269692
		Loss: 0.269692
Iteration: 3000
				Accuracy: 0.851379
		Loss: 0.332691
		Loss: 0.398664
		Loss: 0.398664
Iteration: 4000
				Accuracy: 0.875366
		Loss: 0.314983
		Loss: 0.127959
		Loss: 0.127959
Iteration: 5000
				Accuracy: 0.876404
		Loss: 0.288829
		Loss: 0.268389
		Loss: 0.268389
Iteration: 6000
				Accuracy: 0.872925
		Loss: 0.306257
		Loss: 0.171131
		Loss: 0.171131
Iteration: 7000
				Accuracy: 0.875671
		Loss: 0.312735
		Loss: 0.249351
		Loss: 0.249351
Iteration: 8000
				Accuracy: 0.877319
		Loss: 0.31563
		Loss: 0.285037
		Loss: 0.285036
Iteration: 9000
				Accuracy: 0.871521
		Loss: 0.306121
		Loss: 0.125517
		Loss: 0.125517
Iteration: 10000
				Accuracy: 0.866455
		Loss: 0.352724
		Loss: 0.208356
		Loss: 0.208356
Iteration: 11000
				Accuracy: 0.870239
		Loss: 0.380286
		Loss: 0.195039
		Loss: 0.195039
Iteration: 12000
				Accuracy: 0.862488
		Loss: 0.386133
		Loss: 0.160543
		Loss: 0.160542
Iteration: 13000
				Accuracy: 0.852783
		Loss: 0.395843
		Loss: 0.132938
		Loss: 0.132938
Iteration: 14000
				Accuracy: 0.859985
		Loss: 0.418401
		Loss: 0.0968928
		Loss: 0.0968927
Iteration: 15000
				Accuracy: 0.852112
		Loss: 0.51566
		Loss: 0.159831
		Loss: 0.159831
Iteration: 16000
				Accuracy: 0.863403
		Loss: 0.456039
		Loss: 0.212659
		Loss: 0.212659
Iteration: 17000
				Accuracy: 0.864624
		Loss: 0.401947
		Loss: 0.167202
		Loss: 0.167202
Iteration: 18000
				Accuracy: 0.860229
		Loss: 0.458565
		Loss: 0.194184
		Loss: 0.194184
Iteration: 19000
				Accuracy: 0.857483
		Loss: 0.450797
		Loss: 0.101896
		Loss: 0.101895
Iteration: 20000
				Accuracy: 0.861084
		Loss: 0.50147
		Loss: 0.154803
		Loss: 0.154803
Iteration: 21000
				Accuracy: 0.861511
		Loss: 0.514846
		Loss: 0.181858
		Loss: 0.181857
Iteration: 22000
				Accuracy: 0.853516
		Loss: 0.605593
		Loss: 0.0150778
		Loss: 0.0150775
Iteration: 23000
				Accuracy: 0.85614
		Loss: 0.547736
		Loss: 0.149206
		Loss: 0.149205
Iteration: 24000
				Accuracy: 0.858948
		Loss: 0.55384
		Loss: 0.0264354
		Loss: 0.0264353
Iteration: 25000
				Accuracy: 0.861206
		Loss: 0.583243
		Loss: 0.0411675
		Loss: 0.0411673
Iteration: 26000
				Accuracy: 0.863403
		Loss: 0.5622
		Loss: 0.10747
		Loss: 0.10747
Iteration: 27000
				Accuracy: 0.862976
		Loss: 0.539846
		Loss: 0.0624186
		Loss: 0.0624183
Iteration: 28000
				Accuracy: 0.859436
		Loss: 0.636788
		Loss: 0.0853239
		Loss: 0.0853237
Iteration: 29000
				Accuracy: 0.863831
		Loss: 0.717435
		Loss: 0.0179295
		Loss: 0.0179294
Iteration: 30000
		Loss: 0.0702705
				Accuracy: 0.864563
		Loss: 0.707163
```
</p></details><br/>

<details>
  <summary>Step SGD</summary><p>
  
```
net: "prototxt/train_class_net.prototxt"

# training data count = 2^16
# test data count = 2^14
# test batch size = 2^4
# --> test_iter = 2^10, so we cover all test data per iteration
test_iter: 1024
test_interval: 1000

# Solver method parameters
type: "SGD"
momentum: 0.9
base_lr: 0.01
lr_policy: "step"
gamma: 0.1
stepsize: 7000
max_iter: 21000
# Multiply lr by .1 every 7K steps

# Output info
display: 1000

# snapshot intermediate results
snapshot: 5000
snapshot_prefix: "snapshots"

solver_mode: GPU
```
</p></details><br/>

<details>
  <summary>Accuracy: 0.86</summary><p>
  
```
Iteration: 0
				Accuracy: 0.50177
		Loss: 0.696492
		Loss: 0.691293
		Loss: 0.691293
Iteration: 1000
				Accuracy: 0.701172
		Loss: 0.532814
		Loss: 0.559229
		Loss: 0.559229
Iteration: 2000
				Accuracy: 0.738586
		Loss: 0.515954
		Loss: 0.599992
		Loss: 0.599992
Iteration: 3000
				Accuracy: 0.748413
		Loss: 0.527863
		Loss: 0.5658
		Loss: 0.565801
Iteration: 4000
				Accuracy: 0.726746
		Loss: 0.511349
		Loss: 0.504304
		Loss: 0.504304
Iteration: 5000
				Accuracy: 0.764771
		Loss: 0.494923
		Loss: 0.462972
		Loss: 0.462972
Iteration: 6000
				Accuracy: 0.673767
		Loss: 0.536274
		Loss: 0.555797
		Loss: 0.555797
Iteration: 7000
				Accuracy: 0.672424
		Loss: 0.545039
		Loss: 0.464622
		Loss: 0.464622
Iteration: 8000
				Accuracy: 0.705566
		Loss: 0.531716
		Loss: 0.442604
		Loss: 0.442604
Iteration: 9000
				Accuracy: 0.673645
		Loss: 0.555645
		Loss: 0.502417
		Loss: 0.502417
Iteration: 10000
				Accuracy: 0.631897
		Loss: 0.579053
		Loss: 0.551493
		Loss: 0.551493
Iteration: 11000
				Accuracy: 0.73761
		Loss: 0.478255
		Loss: 0.543852
		Loss: 0.543852
Iteration: 12000
				Accuracy: 0.788574
		Loss: 0.429761
		Loss: 0.389201
		Loss: 0.389201
Iteration: 13000
				Accuracy: 0.838318
		Loss: 0.362347
		Loss: 0.389412
		Loss: 0.389412
Iteration: 14000
				Accuracy: 0.834473
		Loss: 0.371198
		Loss: 0.23972
		Loss: 0.23972
Iteration: 15000
				Accuracy: 0.841858
		Loss: 0.35634
		Loss: 0.245442
		Loss: 0.245442
Iteration: 16000
				Accuracy: 0.85022
		Loss: 0.347115
		Loss: 0.324885
		Loss: 0.324885
Iteration: 17000
				Accuracy: 0.848511
		Loss: 0.351375
		Loss: 0.42697
		Loss: 0.42697
Iteration: 18000
				Accuracy: 0.827576
		Loss: 0.374017
		Loss: 0.370106
		Loss: 0.370106
Iteration: 19000
				Accuracy: 0.85144
		Loss: 0.344632
		Loss: 0.241066
		Loss: 0.241066
Iteration: 20000
				Accuracy: 0.853455
		Loss: 0.345163
		Loss: 0.341853
		Loss: 0.341853
Iteration: 21000
				Accuracy: 0.854309
		Loss: 0.339767
		Loss: 0.346573
		Loss: 0.346573
Iteration: 22000
				Accuracy: 0.85553
		Loss: 0.335848
		Loss: 0.280553
		Loss: 0.280553
Iteration: 23000
				Accuracy: 0.855652
		Loss: 0.335202
		Loss: 0.302192
		Loss: 0.302192
Iteration: 24000
				Accuracy: 0.854858
		Loss: 0.335943
		Loss: 0.403914
		Loss: 0.403914
Iteration: 25000
				Accuracy: 0.857422
		Loss: 0.334265
		Loss: 0.249807
		Loss: 0.249807
Iteration: 26000
				Accuracy: 0.857544
		Loss: 0.333048
		Loss: 0.241089
		Loss: 0.241089
Iteration: 27000
				Accuracy: 0.858459
		Loss: 0.332014
		Loss: 0.214851
		Loss: 0.214851
Iteration: 28000
				Accuracy: 0.856567
		Loss: 0.332509
		Loss: 0.329776
		Loss: 0.329776
Iteration: 29000
				Accuracy: 0.858643
		Loss: 0.333183
		Loss: 0.293083
		Loss: 0.293083
Iteration: 30000
		Loss: 0.207623
				Accuracy: 0.859131
		Loss: 0.332896
```
</p></details><br/>
