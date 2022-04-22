import random

import cv2
import numpy as np
import torch


class HomographyAugmenter:
    """Homography augmentation class"""

    def __init__(
        self,
        crop_hw=None,
        crop_min_scale_factor=0.8,
        crop_max_rotation=np.pi * (15.0 / 180),
        crop_min_distort_factors_xy=(0.6, 0.6),
    ):

        self.crop_height, self.crop_width = crop_hw
        self.crop_min_scale_factor = crop_min_scale_factor
        self.crop_max_rotation = crop_max_rotation
        self.crop_min_distort_factors_xy = crop_min_distort_factors_xy

        self.crop_points = np.float32(
            [
                [0, 0],
                [self.crop_width - 1, 0],
                [0, self.crop_height - 1],
                [self.crop_width - 1, self.crop_height - 1],
            ]
        )

        self.crop_corners = np.array(
            [
                [-(self.crop_width - 1) / 2, -(self.crop_height - 1) / 2],
                [(self.crop_width - 1) / 2, -(self.crop_height - 1) / 2],
                [-(self.crop_width - 1) / 2, (self.crop_height - 1) / 2],
                [(self.crop_width - 1) / 2, (self.crop_height - 1) / 2],
            ]
        ).astype(np.float32)

    def get_min_image_hw(self):
        """Minimum height and width of an image for cropping"""
        # assumes 45 degree rotation
        crop_max_diameter = np.sqrt(self.crop_width ** 2 + self.crop_height ** 2)
        crop_max_diameter += 1
        # TODO incorporate crop_max_rotation
        crop_max_diameter = int(crop_max_diameter + 0.5)
        return crop_max_diameter, crop_max_diameter

    def get_random_homography(self, image_hw, crop_center=None, jitter=None):
        """Generate random homography transformation"""

        image_height, image_width = image_hw
        if jitter is None:
            scale_jitter = torch.rand(1).numpy().astype(np.float32)[0]
            center_jitter = torch.rand(2).numpy().astype(np.float32)
            perspective_jitter = torch.rand(2).numpy().astype(np.float32)
            rotation_jitter = torch.rand(1).numpy().astype(np.float32)[0]
        else:
            scale_jitter = jitter[0]
            center_jitter = jitter[1:3]
            perspective_jitter = jitter[3:5]
            rotation_jitter = jitter[5]

        # decide on scale of the crop. We can only zoom into the crop
        crop_zoom_factor = self.crop_min_scale_factor * scale_jitter + 1 * (
            1 - scale_jitter
        )

        # decide if distorting horizontal or vertical sides, i.e.
        # sides parallel to x-axis or sides parallel to y-axis
        # horver = 0 means distorting top and bottom sides
        # horver = 1 means distorting left and right sides
        horver = np.int32(torch.rand(1).numpy()[0] < 0.5)

        # depending on horver and scale_factor, compute the maximum radius of the crop
        crop_max_radius = 0.5 * np.sqrt(
            (crop_zoom_factor * self.crop_width) ** 2
            + (crop_zoom_factor * self.crop_height) ** 2
        )

        if crop_center is None:
            # decide on crop center
            crop_center = crop_max_radius + center_jitter * (
                np.array([image_width, image_height]) - 2 * crop_max_radius
            )
        else:

            def rnd_sign():
                return 1 if random.random() < 0.5 else -1

            # apply a 10% jitter in pixels to crop center
            crop_center += 0.1 * np.array(
                [self.crop_width * rnd_sign(), self.crop_height * rnd_sign()]
            )

        # decide on scale of the crop's left/top and right/bottom side
        crop_distort_factors = self.crop_min_distort_factors_xy[
            horver
        ] * perspective_jitter + 1 * (1 - perspective_jitter)

        # decide on crop rotation
        rotation_jitter = 2 * rotation_jitter - 1
        rotation_jitter *= self.crop_max_rotation
        cosa = np.cos(rotation_jitter)
        sina = np.sin(rotation_jitter)

        # zoom into crop
        scaled_crop_corners = self.crop_corners.copy() * crop_zoom_factor

        # perspective distort
        if horver == 0:
            # distort in x-axis
            scaled_crop_corners[0, 0] *= crop_distort_factors[0]
            scaled_crop_corners[1, 0] *= crop_distort_factors[0]
            scaled_crop_corners[2, 0] *= crop_distort_factors[1]
            scaled_crop_corners[3, 0] *= crop_distort_factors[1]
        else:
            # distort in y-axis
            scaled_crop_corners[0, 1] *= crop_distort_factors[0]
            scaled_crop_corners[2, 1] *= crop_distort_factors[0]
            scaled_crop_corners[1, 1] *= crop_distort_factors[1]
            scaled_crop_corners[3, 1] *= crop_distort_factors[1]

        # rotate crop corners
        rotated_crop_corners = scaled_crop_corners.copy()
        rotated_crop_corners[:, 0] = (
            cosa * scaled_crop_corners[:, 0] - sina * scaled_crop_corners[:, 1]
        )
        rotated_crop_corners[:, 1] = (
            sina * scaled_crop_corners[:, 0] + cosa * scaled_crop_corners[:, 1]
        )

        # shift crop center
        image_points = np.float32(crop_center + rotated_crop_corners)

        # make source and destination points
        cv_image_to_crop = cv2.getPerspectiveTransform(image_points, self.crop_points)
        cv_crop_to_image = cv2.getPerspectiveTransform(self.crop_points, image_points)

        # in pytorch crop to image
        hmats_toimage = np.array(cv_image_to_crop, dtype=np.float32)
        # in pytorch image to crop
        hmats_tocrop = np.array(cv_crop_to_image, dtype=np.float32)

        return (
            cv_image_to_crop,
            cv_crop_to_image,
            hmats_toimage,
            hmats_tocrop,
            crop_center,
        )

    @staticmethod
    def _is_valid_kpt(kpt, h, w):
        return (0 <= kpt[0] <= h - 1) and (0 <= kpt[1] <= w - 1)

    def warp_image_and_kpts(self, image, kpts, cv_image_to_crop):
        # convert to ndarray from PIL
        image = np.array(image)
        image_w = cv2.warpPerspective(
            image, cv_image_to_crop, (self.crop_height, self.crop_width)
        )

        kpts_w = torch.matmul(torch.FloatTensor(cv_image_to_crop), kpts.t()).t()
        kpts_w = [kpt / kpt[-1] for kpt in kpts_w]
        kpts_w = torch.stack(kpts_w)

        mask = torch.BoolTensor(
            [
                self._is_valid_kpt(kpt, self.crop_height, self.crop_width)
                for kpt in kpts_w
            ]
        )
        return image_w, kpts_w[mask]

    def warp_image(self, image, cv_image_to_crop, target_hw=None):
        """Perspective warp image to crop size"""
        if target_hw is None:
            target_hw = (self.crop_height, self.crop_width)

        # convert to ndarray from PIL
        image = np.array(image)
        result = cv2.warpPerspective(
            image, cv_image_to_crop, (target_hw[1], target_hw[0])
        )
        return result
