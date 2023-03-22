# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Image processor class for UnifiedIO.
"""

from typing import Sequence, Tuple, Union

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_transforms import to_channel_dimension_format

BIAS = np.array([0.485, 0.456, 0.406])
SCALE = np.array([0.229, 0.224, 0.225])
IMAGE_INPUT_SIZE = [384, 384]
IMAGE_INPUT_PATCH_SIZE = 16
IMAGE_TARGET_SIZE = [256, 256]


class UioImageProcessor(BaseImageProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def resize(
        self,
        image: np.ndarray,
        target_size: Sequence[int],
        mode: Union[str, InterpolationMode] = "bilinear",
        antialias=True,
    ):
        if isinstance(mode, str):
            mode = InterpolationMode(mode)
        if image.dtype == np.uint8:
            image = image / 255.0
        image = F.resize(
            torch.as_tensor(image.transpose((2, 0, 1))),
            target_size,
            antialias=antialias,
            interpolation=mode,
        )
        image = np.transpose(image.numpy().astype(np.float32), [1, 2, 0])
        return image

    def normalize_image(self, image) -> np.ndarray:
        """Pixel normalizing used by UnifiedIO"""
        image -= BIAS.reshape((1, 1, 3))
        image /= SCALE.reshape((1, 1, 3))
        return image

    def resize_and_pad(self, image: np.ndarray, size) -> Tuple[np.ndarray, np.ndarray]:
        """Resize and pad `image` to `size` and returns a mask over pixels introduced
        by padding
        """
        h, w = image.shape[:2]
        scale = size[0] / max(h, w)
        if scale != 1.0:
            scale_to = (int(h * scale), int(w * scale))
            image = self.resize(image, scale_to)
        else:
            scale_to = (h, w)
        image_mask = np.zeros(size, dtype=bool)
        image_mask[: scale_to[0], : scale_to[1]] = True
        image = np.pad(
            image, [[0, size[0] - scale_to[0]], [0, size[1] - scale_to[1]], [0, 0]]
        )
        return image, image_mask

    def preprocess_image(
        self, input_image, mask_region=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess an image for processing UnifiedIO

        :param input_image: image array in [h, w, 3] in float or uint8 format
        :param mask_region: Optional region to include in the image mask, used for
                image inpaintin
        :return: preprocessed image and image-patch mask
        """
        n_patches = 384 // 16
        if input_image is not None:
            original_size = input_image.shape
            input_image, image_mask = self.resize_and_pad(input_image, IMAGE_INPUT_SIZE)

            if mask_region is not None:
                region = (
                    mask_region / max(original_size[:2]) * max(input_image.shape[:2])
                )
                x1, y1, x2, y2 = np.round(region).astype(np.int32)
                region_mask = np.ones_like(image_mask)
                region_mask[y1:y2, x1:x2] = 0
                image_mask = image_mask * region_mask

            # Convert mask over pixels to mask of image patches
            image_mask = self.resize(
                np.expand_dims(image_mask, 2),
                [n_patches, n_patches],
                InterpolationMode.NEAREST,
                antialias=False,
            )
            image_mask = image_mask.reshape((-1,)).astype(np.int32)
        else:
            if mask_region is not None:
                raise ValueError()
            # Masked, dummy values since this code does not support skipping the image
            input_image = np.zeros((384, 384, 3), np.float32)
            image_mask = np.zeros((n_patches * n_patches,), dtype=np.int32)
        input_image = self.normalize_image(input_image)
        input_image = to_channel_dimension_format(input_image, "channels_first")
        return input_image, image_mask

    def preprocess(self, images, mask_regions = None):
        """Preprocess a batch of images for processing UnifiedIO

        :param images: list of image arrays in [h, w, 3] in float or uint8 format
        :return: preprocessed image and image-patch mask
        """
        input_images = []
        image_masks = []
        for idx, image in enumerate(images):
            input_image, image_mask = self.preprocess_image(
                image, None if mask_regions is None else mask_regions[idx]
            )
            input_images.append(input_image)
            image_masks.append(image_mask)

        input_images = np.stack(input_images)
        image_masks = np.stack(image_masks)

        return input_images, image_masks
