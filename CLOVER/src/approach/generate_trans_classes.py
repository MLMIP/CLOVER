import copy

import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

import torch
import numpy as np
import matplotlib.pyplot as plt


# 假设 x 是形状为 (3, 224, 224) 的图像
def show_image(x, name):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()  # 转换为 NumPy 数组
    x = np.transpose(x, (1, 2, 0))  # 调整维度顺序 -> (224, 224, 3)
    plt.imshow(x)
    plt.axis('off')  # 去除坐标轴
    # plt.show()
    plt.savefig(f'/kaijiang/yuzhang/projects/SEED/imgs/debug/{name}.png')


# 在 Debug 窗口调用 show_image(tensor)


# 生成 3x3 高斯模糊卷积核（Sigma=0.5）
def gaussian_kernel(size=3, sigma=0.5):
    ax = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def generate_trans_classes(exp_mode, images, targets):
    """
    Generate translation classes for all languages.
    """

    raw_images = copy.deepcopy(images)
    size = raw_images.shape[1:]

    if exp_mode in ['17.1', '18.1', '19.1']:
        """blurring3305"""
        kernel = gaussian_kernel(size=3, sigma=0.5)
        # 扩展为 (3,1,3,3) 适配 RGB 通道
        kernel = kernel.repeat(3, 1, 1, 1)  # 变成 (3, 1, 3, 3)
        kernel = kernel.to(images.device)  # 保持相同设备 (CPU/GPU)
        images = torch.stack([images,
                              F.conv2d(images, kernel, padding=1, groups=3)
                              ], 1)
    elif exp_mode in ['17.2', '18.2', '19.2']:
        """blurring5510"""
        kernel = gaussian_kernel(size=5, sigma=1)
        # 扩展为 (3,1,3,3) 适配 RGB 通道
        kernel = kernel.repeat(3, 1, 1, 1)  # 变成 (3, 1, 3, 3)
        kernel = kernel.to(images.device)  # 保持相同设备 (CPU/GPU)
        images = torch.stack([images,
                              F.conv2d(images, kernel, padding=2, groups=3)
                              ], 1)
    elif exp_mode in ['17.3', '18.3', '19.3']:
        """blurring7720"""
        kernel = gaussian_kernel(size=7, sigma=2)
        # 扩展为 (3,1,3,3) 适配 RGB 通道
        kernel = kernel.repeat(3, 1, 1, 1)  # 变成 (3, 1, 3, 3)
        kernel = kernel.to(images.device)  # 保持相同设备 (CPU/GPU)
        images = torch.stack([images,
                              F.conv2d(images, kernel, padding=3, groups=3)
                              ], 1)
    elif exp_mode in ['17.4', '18.4', '19.4']:
        """centerCrop192"""
        transform = transforms.Compose([
            transforms.CenterCrop(192),  # 裁剪
            transforms.Resize(224)  # 缩放回 224x224
        ])

        images = torch.stack([images,
                              torch.stack([transform(image) for image in images])
                              ], 1)
    elif exp_mode in ['17.5', '18.5', '19.5']:
        """centerCrop160"""
        transform = transforms.Compose([
            transforms.CenterCrop(160),  # 裁剪
            transforms.Resize(224)  # 缩放回 224x224
        ])
        images = torch.stack([images,
                              torch.stack([transform(image) for image in images])
                              ], 1)
    elif exp_mode in ['17.6', '18.6', '19.6']:
        """centerCrop128"""
        transform = transforms.Compose([
            transforms.CenterCrop(128),  # 裁剪
            transforms.Resize(224)  # 缩放回 224x224
        ])
        images = torch.stack([images,
                              torch.stack([transform(image) for image in images])
                              ], 1)
    elif exp_mode in ['17.7', '18.7', '19.7']:
        """centerCrop96"""
        transform = transforms.Compose([
            transforms.CenterCrop(96),  # 裁剪
            transforms.Resize(224)  # 缩放回 224x224
        ])
        images = torch.stack([images,
                              torch.stack([transform(image) for image in images])
                              ], 1)
    elif exp_mode in ['17.8', '18.8', '19.8']:
        """resize256"""
        transform = transforms.Compose([
            transforms.Resize(256)  # 缩放
        ])
        images = torch.stack([images,
                              torch.stack([transform(image) for image in images])
                              ], 1)
    elif exp_mode in ['17.9', '18.9', '19.9']:
        """resize288"""
        transform = transforms.Compose([
            transforms.Resize(288)  # 缩放
        ])
        images = torch.stack([images,
                              torch.stack([transform(image) for image in images])
                              ], 1)
    elif exp_mode in ['17.10', '18.10', '19.10']:
        """resize320"""
        transform = transforms.Compose([
            transforms.Resize(320)  # 缩放
        ])
        images = torch.stack([images,
                              torch.stack([transform(image) for image in images])
                              ], 1)
    elif exp_mode in ['17.11', '18.11']:
        """GBR"""
        images = torch.stack([images,
                              torch.stack([images[:, 1, :, :], images[:, 2, :, :], images[:, 0, :, :]], 1)
                              ], 1)
    elif exp_mode in ['17.12', '18.12']:
        """BRG"""
        images = torch.stack([images,
                              torch.stack([images[:, 2, :, :], images[:, 0, :, :], images[:, 1, :, :]], 1)
                              ], 1)
    elif exp_mode in ['17.13', '18.13']:
        """noise05"""
        # 生成与输入数据同尺寸的高斯噪声，均值为0，标准差为noise_std
        noise = torch.randn_like(images) * 0.5
        images = torch.stack([images,
                              images + noise
                              ], 1)
    elif exp_mode in ['17.14', '18.14']:
        """noise10"""
        # 生成与输入数据同尺寸的高斯噪声，均值为0，标准差为noise_std
        noise = torch.randn_like(images) * 1.0
        images = torch.stack([images,
                              images + noise
                              ], 1)
    elif exp_mode in ['17.15', '18.15']:
        """noise15"""
        # 生成与输入数据同尺寸的高斯噪声，均值为0，标准差为noise_std
        noise = torch.randn_like(images) * 1.5
        images = torch.stack([images,
                              images + noise
                              ], 1)
    elif exp_mode in ['18.16']:
        "rot90"
        images = torch.stack([images,
                              torch.rot90(images, 1, (2, 3))
                              ], 1)
    elif exp_mode in ['18.17']:
        "rot180"
        images = torch.stack([images,
                              torch.rot90(images, 2, (2, 3))
                              ], 1)
    elif exp_mode in ['18.18']:
        "rot270"
        images = torch.stack([images,
                              torch.rot90(images, 3, (2, 3))
                              ], 1)
    elif exp_mode in ['20.19']:
        "3color"
        images = torch.stack([images,
                              torch.stack([raw_images[:, 0, :, :], raw_images[:, 2, :, :], raw_images[:, 1, :, :]], 1),
                              torch.stack([raw_images[:, 1, :, :], raw_images[:, 0, :, :], raw_images[:, 2, :, :]], 1),
                              torch.stack([raw_images[:, 2, :, :], raw_images[:, 1, :, :], raw_images[:, 0, :, :]], 1),
                              ], 1)


    # show_image(images[0][0], 'raw')
    # show_image(images[0][1], 'trans')

    if "21.1" in exp_mode:
        """color+noise"""
        # 生成与输入数据同尺寸的高斯噪声，均值为0，标准差为noise_std
        noise = torch.randn_like(raw_images) * 0.5

        images = torch.stack([images,
                              raw_images + noise,
                              torch.stack([raw_images[:, 1, :, :], raw_images[:, 2, :, :], raw_images[:, 0, :, :]], 1),
                              torch.stack([raw_images[:, 2, :, :], raw_images[:, 0, :, :], raw_images[:, 1, :, :]], 1)
                              ], 1)

        images = images.view(-1, *size)
        targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)

    elif "20.19" in exp_mode:
        """cdssl+moreColor"""
        # 生成与输入数据同尺寸的高斯噪声，均值为0，标准差为noise_std
        noise = torch.randn_like(raw_images) * 0.5

        images = torch.stack([images[:, 0, :, :],
                              images[:, 1, :, :],
                              images[:, 2, :, :],
                              images[:, 3, :, :],
                              raw_images + noise,
                              torch.rot90(raw_images, 1, (2, 3)),
                              torch.rot90(raw_images, 2, (2, 3)),
                              torch.rot90(raw_images, 3, (2, 3)),
                              torch.stack([raw_images[:, 1, :, :], raw_images[:, 2, :, :], raw_images[:, 0, :, :]], 1),
                              torch.stack([raw_images[:, 2, :, :], raw_images[:, 0, :, :], raw_images[:, 1, :, :]], 1)
                              ], 1)

        images = images.view(-1, *size)
        targets = torch.stack([targets * 10 + k for k in range(10)], 1).view(-1)
    elif "17." in exp_mode:
        """SSL+"""
        images = torch.stack([images[:, 0, :, :],
                              images[:, 1, :, :],
                              torch.rot90(raw_images, 1, (2, 3)),
                              torch.rot90(raw_images, 2, (2, 3)),
                              torch.rot90(raw_images, 3, (2, 3)),
                              ], 1)
        images = images.view(-1, *size)
        targets = torch.stack([targets * 5 + k for k in range(5)], 1).view(-1)
    elif "18." in exp_mode:
        """Baseline+"""
        images = images.view(-1, *size)
        targets = torch.stack([targets * 2 + k for k in range(2)], 1).view(-1)
    elif "19." in exp_mode:
        """CDSSL+"""
        # 生成与输入数据同尺寸的高斯噪声，均值为0，标准差为noise_std
        noise = torch.randn_like(raw_images) * 0.5

        images = torch.stack([images[:, 0, :, :],
                              images[:, 1, :, :],
                              raw_images + noise,
                              torch.rot90(raw_images, 1, (2, 3)),
                              torch.rot90(raw_images, 2, (2, 3)),
                              torch.rot90(raw_images, 3, (2, 3)),
                              torch.stack([raw_images[:, 1, :, :], raw_images[:, 2, :, :], raw_images[:, 0, :, :]], 1),
                              torch.stack([raw_images[:, 2, :, :], raw_images[:, 0, :, :], raw_images[:, 1, :, :]], 1)
                              ], 1)

        images = images.view(-1, *size)
        targets = torch.stack([targets * 8 + k for k in range(8)], 1).view(-1)


    return images, targets
