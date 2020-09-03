import numpy as np
import time
from keras import backend as K


def augment_dataset(images, masks, segs, aug_fn_arg):
    start_augment_time = time.time()

    num_images = len(images)
    aug_fn = aug_fn_arg[0]
    aug_arg = aug_fn_arg[1]

    augmented_images = np.zeros_like(images)
    augmented_masks = np.zeros_like(masks)
    augmented_segs = np.zeros_like(segs)

    for i in range(num_images):
        image = images[i, :, :]
        mask = masks[i, :, :]
        seg = segs[i, :, :]
        augmented_images[i, :, :], augmented_masks[i, :, :], augmented_segs[i, :, :], _, _ \
            = aug_fn(image, mask, seg, aug_arg)

    aug_desc = aug_fn(None, None, None, aug_arg, True)

    end_augment_time = time.time()
    total_aug_time = end_augment_time - start_augment_time

    return [augmented_images, augmented_masks, augmented_segs, aug_desc, total_aug_time]


def no_aug(image, mask, seg, aug_args, desc_only=False, sample_ind=None, set=None):
    desc = "no aug"
    if desc_only is False:
        return image, mask, seg, desc, 0
    else:
        return desc


def flip_aug(image, mask, seg, aug_args, desc_only=False, sample_ind=None, set=None):
    start_augment_time = time.time()

    flip_type = aug_args['flip_type']

    if flip_type == 'up-down':
        axis = 1
    elif flip_type == 'left-right':
        axis = 0

    aug_desc = "flip aug: " + flip_type

    if desc_only is False:
        aug_image = np.flip(image, axis=axis)
        if mask is not None:
            aug_mask = np.flip(mask, axis=axis)
        else:
            aug_mask = None
        if seg is not None:
            aug_seg = np.flip(seg, axis=axis)
        else:
            aug_seg = None

        end_augment_time = time.time()
        augment_time = end_augment_time - start_augment_time

        return aug_image, aug_mask, aug_seg, aug_desc, augment_time
    else:
        return aug_desc


def gaussian_noise_aug(image, mask, seg, aug_args, desc_only=False, sample_ind=None, set=None):
    start_augment_time = time.time()

    var = aug_args['variance']

    if var == 'random':
        min_var = aug_args['min']
        max_var = aug_args['max']

        var = np.random.uniform(min_var, max_var)

        aug_desc = "gaussian noise aug (variance_random" + ", min_" + str(min_var) + ", max_" + str(max_var) + ")"
    else:
        aug_desc = "gaussian noise aug (variance_" + str(var) + ")"

    if desc_only is True:
        return aug_desc
    else:
        mean = 0
        sigma = var**0.5

        if K.image_dim_ordering() == 'tf':
            width, height, ch = image.shape
            gauss = np.random.normal(mean, sigma, (width, height, ch))
            gauss = gauss.reshape(width, height, ch)
        else:
            ch, width, height = image.shape
            gauss = np.random.normal(mean, sigma, (ch, width, height))
            gauss = gauss.reshape(ch, width, height)

        aug_image = image + gauss
        aug_image[aug_image > 255] = 255
        aug_image[aug_image < 0] = 0

        aug_mask = mask
        aug_segs = seg

        end_augment_time = time.time()
        augment_time = end_augment_time - start_augment_time

        return aug_image, aug_mask, aug_segs, aug_desc, augment_time


def combo_aug(image, mask, segs, aug_fn_args, desc_only=False, sample_ind=None, set=None):
    start_augment_time = time.time()

    aug_desc = "combo_aug ("

    aug_image = image
    aug_mask = mask
    aug_segs = segs

    for aug_fn_arg_ind in range(len(aug_fn_args)):

        aug_fn_arg = aug_fn_args[aug_fn_arg_ind]

        aug_fn = aug_fn_arg[0]
        aug_arg = aug_fn_arg[1]

        if desc_only is True:
            aug_desc_part = aug_fn(aug_image, aug_mask, aug_segs, aug_arg, desc_only=desc_only, sample_ind=sample_ind, set=set)
        else:
            aug_image, aug_mask, aug_segs, aug_desc_part, _ = aug_fn(aug_image, aug_mask, aug_segs, aug_arg, desc_only=desc_only, sample_ind=sample_ind, set=set)

        if aug_fn_arg_ind != 0:
            aug_desc += ", "

        aug_desc += aug_desc_part

    aug_desc += ")"

    end_augment_time = time.time()
    augment_time = end_augment_time - start_augment_time

    if desc_only is True:
        return aug_desc
    else:
        return aug_image, aug_mask, aug_segs, aug_desc, augment_time


def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))
