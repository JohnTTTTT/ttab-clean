# -*- coding: utf-8 -*-
import random

import torch
import torchvision.transforms as transforms

__imagenet_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
__imagenet_pca = {
    "eigval": torch.Tensor([0.2175, 0.0188, 0.0045]),
    "eigvec": torch.Tensor(
        [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ]
    ),
}


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [transforms.CenterCrop(input_size), transforms.ToTensor()]
    if normalize is not None:
        t_list += [transforms.Normalize(**normalize)]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list
    return transforms.Compose(t_list)


def scale_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [transforms.RandomCrop(input_size), transforms.ToTensor()]
    if normalize is not None:
        t_list += [transforms.Normalize(**normalize)]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list
    return transforms.Compose(t_list)


def pad_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    padding = int((scale_size - input_size) / 2)
    t_list = [
        transforms.RandomCrop(input_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if normalize is not None:
        t_list += [transforms.Normalize(**normalize)]
    return transforms.Compose(t_list)


def inception_preproccess(input_size, normalize=__imagenet_stats, scale=None):
    t_list = [
        transforms.RandomResizedCrop(input_size, scale=scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if normalize is not None:
        t_list += [transforms.Normalize(**normalize)]
    return transforms.Compose(t_list)


def inception_color_preproccess(input_size, normalize=__imagenet_stats):
    t_list = [
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        Lighting(0.1, __imagenet_pca["eigval"], __imagenet_pca["eigvec"]),
    ]
    if normalize is not None:
        t_list += [transforms.Normalize(**normalize)]
    return transforms.Compose(t_list)


# TODO: improve transform api.
def get_transform(
    name="imagenet",
    input_size=None,
    scale_size=None,
    normalize=None,
    augment=True,
    color_process=False,
):
    normalize = normalize or __imagenet_stats

    if "imagenet" in name:
        scale_size = scale_size or (36 if "downsampled" in name else 256)
        input_size = input_size or (32 if "downsampled" in name else 224)

        if augment:
            if color_process:
                preprocess_fn = inception_color_preproccess
            else:
                preprocess_fn = inception_preproccess
            return preprocess_fn(input_size, normalize=normalize)
        else:
            return scale_crop(
                input_size=input_size, scale_size=scale_size, normalize=normalize
            )
    elif "cifar" in name:
        input_size = input_size or 32
        if input_size == 32:
            if augment:
                scale_size = scale_size or 40
                return pad_random_crop(
                    input_size, scale_size=scale_size, normalize=normalize
                )
            else:
                scale_size = scale_size or 32
                return scale_crop(
                    input_size=input_size, scale_size=scale_size, normalize=normalize
                )
        elif input_size > 32:
            # resize to a larger image, used in vit.
            if augment:
                return inception_preproccess(
                    input_size, normalize=normalize, scale=(0.05, 1)
                )
            else:
                scale_size = scale_size or 32
                return transforms.Compose(
                    [
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(**normalize),
                    ]
                )
    elif name == "mnist":
        normalize = {"mean": [0.5], "std": [0.5]}
        input_size = input_size or 28
        if augment:
            scale_size = scale_size or 32
            return pad_random_crop(
                input_size, scale_size=scale_size, normalize=normalize
            )
        else:
            scale_size = scale_size or 32
            return scale_crop(
                input_size=input_size, scale_size=scale_size, normalize=normalize
            )
    elif name in ["officehome", "pacs"]:
        if augment:
            transform = transforms.Compose(
                [
                    # transforms.Resize((224,224)),
                    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        return transform
    elif name == "custom_dataset":
        # default transform, for user-provided datasets.
        if augment:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor(), normalize])
        return transform
    elif name == "affectnet":
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        if augment:
            # FMAE-IAT train-time pipeline
            transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    (input_size, input_size),
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.RAND),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
            ])
        else:
            # FMAE-IAT eval-time pipeline
            transform = transforms.Compose(
            [transforms.Resize([224, 224]),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)]
        )

        return transform
    else:
        raise NotImplementedError


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """Composes several transforms together in random order."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))
