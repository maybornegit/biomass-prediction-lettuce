import numpy as np
import random, tqdm, torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance

class LettuceDataset(Dataset):
    def __init__(self, rgbd_set, transform=None):
        self.transform = transform
        self.dataset_rgbd = rgbd_set

    def __len__(self):
        return len(self.dataset_rgbd)

    def __getitem__(self, index):
        data_info = self.dataset_rgbd[index]
        rgbd_image = data_info[0].copy()
        rgbd_image = np.transpose(rgbd_image, (2, 0, 1))  # Assuming rgbd_image is [height, width, channels]
        rgbd = torch.from_numpy(rgbd_image).float()  # Convert to PyTorch tensor
        true_biomass = data_info[1]
        return rgbd, true_biomass
class AugmentationTransform:
    def __call__(self, rgbd_image, aug_flip,aug_rot, aug_bright):
        augmented_images = []

        # Rotations
        if aug_rot:
            ninety = np.rot90(rgbd_image.copy())
            augmented_images.append(ninety)

            oneeighty = np.rot90(ninety.copy())
            augmented_images.append(oneeighty)

            twoseventy = np.rot90(oneeighty.copy())
            augmented_images.append(twoseventy)

        # Flips
        if aug_flip:
            flip_hor = np.fliplr(rgbd_image.copy())
            augmented_images.append(flip_hor)

            flip_vert = np.flipud(rgbd_image.copy())
            augmented_images.append(flip_vert)

        #Brightness changes
        if aug_bright != 0:
            brightness_factors = [0.8, 1.2, 0.9, 1.2]
            brightness_factors = brightness_factors[:aug_bright]
            stop_itr = len(augmented_images)
            if rgbd_image.shape[2] == 4:
                for img in augmented_images[:stop_itr]:  # Only apply brightness to original and rotations
                    for factor in brightness_factors:
                        augmented_img = img.copy().astype(np.uint8)
                        rgb_part = augmented_img[:, :, :3]
                        im = Image.fromarray(rgb_part)
                        enhancer = ImageEnhance.Brightness(im)
                        im = enhancer.enhance(factor)
                        augmented_img[:, :, :3] = np.array(im)
                        augmented_images.append(augmented_img)
            elif rgbd_image.shape[2] == 8:
                for img in augmented_images[:stop_itr]:  # Only apply brightness to original and rotations
                    rgbds = [img[:, :, :4], img[:, :, 4:]]
                    for factor in brightness_factors:
                        new_imgs = []
                        for rgbd in rgbds:
                            augmented_img = rgbd.copy().astype(np.uint8)
                            rgb_part = augmented_img[:, :, :3]
                            im = Image.fromarray(rgb_part)
                            enhancer = ImageEnhance.Brightness(im)
                            im = enhancer.enhance(factor)
                            augmented_img[:, :, :3] = np.array(im)
                            new_imgs.append(augmented_img)
                        combined_img = np.concatenate((new_imgs[0],new_imgs[1]),axis=-1)
                        augmented_images.append(combined_img)

        augmented_images.append(rgbd_image)
        augmented_images = [np.ascontiguousarray(aug) for aug in augmented_images]
        return augmented_images

def data_augmentations(rgbdTrue, train_split, aug_flip,aug_rot, aug_bright):
    training = list(range(len(rgbdTrue)))
    random.shuffle(training)
    train = training[:int(train_split * len(rgbdTrue))]
    valid = training[int(train_split * len(rgbdTrue)):]
    augmentation = AugmentationTransform()

    augmentSet = [rgbdTrue[idx] for idx in valid]
    augmentSet_valid = []
    for k in tqdm.tqdm(range(len(augmentSet))):
        augment = augmentation(augmentSet[k][0], aug_flip,aug_rot, aug_bright)
        [augmentSet_valid.append((augment[x], augmentSet[k][1])) for x in range(len(augment))]
    #
    augmentSet = [rgbdTrue[idx] for idx in train]
    augmentSet_train = []
    for k in tqdm.tqdm(range(len(augmentSet))):
        augment = augmentation(augmentSet[k][0], aug_flip,aug_rot, aug_bright)
        [augmentSet_train.append((augment[x], augmentSet[k][1])) for x in range(len(augment))]

    return augmentSet_train, augmentSet_valid
