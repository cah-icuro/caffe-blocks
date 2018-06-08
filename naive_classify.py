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
