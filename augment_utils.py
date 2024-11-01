import numpy as np
import random, tqdm, torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import torch
from torchvision.transforms import v2

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

    def get_lighting(self, index):
        return self.dataset_rgbd[index][2]

class AugmentationTransform:
    def __call__(self, rgbd_image, aug_flip,aug_rot, aug_bright, aug_crop, aug_grey):
        augmented_images = []
        input = Image.fromarray((rgbd_image * 255).astype('uint8'))
        # Rotations
        rotation_transform = v2.RandomRotation(degrees=180)
        crop = v2.RandomResizedCrop(size=(640, 640), antialias=True)
        Hflip = v2.RandomHorizontalFlip(p=1)
        Vflip = v2.RandomVerticalFlip(p=1)
        color = v2.ColorJitter((0.5,1.5),(0.5,1.5),(0.5,1.5),(-0.5,0.5))
        grey = v2.RandomGrayscale(p=0.1)

        [augmented_images.append(np.array(rotation_transform(input))) for _ in range(aug_rot)]
        [augmented_images.append(np.array(crop(input))) for _ in range(aug_crop)]
        if aug_flip:
            augmented_images.append(np.array(Hflip(input)))
            augmented_images.append(np.array(Vflip(input)))
            augmented_images.append(np.array(Vflip(Hflip(input))))

        new_imgs = []
        for im in augmented_images:
            for _ in range(aug_bright):
                input = Image.fromarray(((im[:,:,:3]) * 255).astype('uint8'))
                new_imgs.append(np.concatenate((np.array(color(input)),im[:,:,3].reshape((640,640,1))),axis=2))
        if aug_grey:
            for i, im in enumerate(new_imgs):
                input = Image.fromarray(((im[:, :, :3]) * 255).astype('uint8'))
                new_imgs[i] = np.concatenate((np.array(grey(input)), im[:, :, 3].reshape((640,640,1))), axis=2)

        # if aug_rot:
        #
        #     for i in range(aug_rot):
        #         input = Image.fromarray((rgbd_image*255).astype('uint8'))
        #         r1 = np.array(rotation_transform(input))
        #         augmented_images.append(r1)
        #     # ninety = np.rot90(rgbd_image.copy())
        #     # augmented_images.append(ninety)
        #     #
        #     # oneeighty = np.rot90(ninety.copy())
        #     # augmented_images.append(oneeighty)
        #     #
        #     # twoseventy = np.rot90(oneeighty.copy())
        #     # augmented_images.append(twoseventy)
        #
        # # Flips
        # if aug_flip:
        #     flip_hor = np.fliplr(rgbd_image.copy())
        #     augmented_images.append(flip_hor)
        #
        #     flip_vert = np.flipud(rgbd_image.copy())
        #     augmented_images.append(flip_vert)
        #
        # #Brightness changes
        # if aug_bright != 0:
        #     brightness_factors = [0.8, 1.2, 0.9, 1.1]
        #     brightness_factors = brightness_factors[:aug_bright]
        #     stop_itr = len(augmented_images)
        #     if rgbd_image.shape[2] == 4:
        #         for img in augmented_images[:stop_itr]:  # Only apply brightness to original and rotations
        #             for factor in brightness_factors:
        #                 augmented_img = img.copy().astype(np.uint8)
        #                 rgb_part = augmented_img[:, :, :3]
        #                 im = Image.fromarray(rgb_part)
        #                 enhancer = ImageEnhance.Brightness(im)
        #                 im = enhancer.enhance(factor)
        #                 augmented_img[:, :, :3] = np.array(im)
        #                 augmented_images.append(augmented_img)
        #     elif rgbd_image.shape[2] == 8:
        #         for img in augmented_images[:stop_itr]:  # Only apply brightness to original and rotations
        #             rgbds = [img[:, :, :4], img[:, :, 4:]]
        #             for factor in brightness_factors:
        #                 new_imgs = []
        #                 for rgbd in rgbds:
        #                     augmented_img = rgbd.copy().astype(np.uint8)
        #                     rgb_part = augmented_img[:, :, :3]
        #                     im = Image.fromarray(rgb_part)
        #                     enhancer = ImageEnhance.Brightness(im)
        #                     im = enhancer.enhance(factor)
        #                     augmented_img[:, :, :3] = np.array(im)
        #                     new_imgs.append(augmented_img)
        #                 combined_img = np.concatenate((new_imgs[0],new_imgs[1]),axis=-1)
        #                 augmented_images.append(combined_img)

        new_imgs.append(rgbd_image)
        augmented_images = [np.ascontiguousarray(aug) for aug in new_imgs]
        return augmented_images

def data_augmentations(rgbdTrue, train_split, aug_flip,aug_rot, aug_bright, aug_crop, aug_grey):
    training = list(range(len(rgbdTrue)))
    random.shuffle(training)
    train = training[:int(train_split * len(rgbdTrue))]
    valid = training[int(train_split * len(rgbdTrue)):]
    augmentation = AugmentationTransform()

    # augmentSet = [rgbdTrue[idx] for idx in valid]
    # augmentSet_valid = []
    # for k in tqdm.tqdm(range(len(augmentSet))):
    #     augment = augmentation(augmentSet[k][0], aug_flip,aug_rot, aug_bright, aug_crop, aug_grey)
    #     [augmentSet_valid.append((augment[x], augmentSet[k][1])) for x in range(len(augment))]

    augmentSet = [rgbdTrue[idx] for idx in train]
    augmentSet_train = []
    for k in tqdm.tqdm(range(len(augmentSet))):
        augment = augmentation(augmentSet[k][0], aug_flip,aug_rot, aug_bright, aug_crop, aug_grey)
        [augmentSet_train.append((augment[x], augmentSet[k][1])) for x in range(len(augment))]

    return augmentSet_train, valid
