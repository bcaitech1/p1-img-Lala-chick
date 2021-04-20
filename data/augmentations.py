import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms_distortion(cfg):
    return A.Compose([
        A.CenterCrop(cfg['img_size'], cfg['img_size'], p=1.),
        A.HueSaturationValue(),
        A.OneOf([
                A.OpticalDistortion(p=0.4),
                A.GridDistortion(p=0.2),
                A.IAAPiecewiseAffine(p=0.4),
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit = (-0.1, 0.1), p = 0.5),
        A.Normalize(mean=[0.56019358, 0.52410121, 0.501457], std=[0.23318603, 0.24300033, 0.24567522], max_pixel_value=255.0, p = 1.0),
        ToTensorV2(p=1.0),
    ])

def get_train_transforms_no_distortion(cfg):
    return A.Compose([
        A.CenterCrop(cfg['img_size'], cfg['img_size'], p=1.),
        A.HueSaturationValue(),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit = (-0.1, 0.1), p = 0.5),
        A.Normalize(mean=[0.56019358, 0.52410121, 0.501457], std=[0.23318603, 0.24300033, 0.24567522], max_pixel_value=255.0, p = 1.0),
        ToTensorV2(p=1.0),
    ])

def get_test_transforms(img_size):
    return A.Compose([
        A.CenterCrop(img_size, img_size, p=1.),
        A.Normalize(mean=[0.56019358, 0.52410121, 0.501457], std=[0.23318603, 0.24300033, 0.24567522], max_pixel_value=255.0, p = 1.0),
        ToTensorV2(p=1.0),
    ])
