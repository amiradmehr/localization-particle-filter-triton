#!/usr/bin/env python3

import cv2
import numpy as np
import yaml


class LikelihoodField:

    def __init__(self) -> None:

        # Path to the map image file
        self.map_yaml_path = "imgs/map.yaml"
        self.lookup_table = None
        self.grey_pixels = None

        # Read map yaml file
        with open(self.map_yaml_path, 'r') as stream:
            try:
                self.map_metadata = yaml.safe_load(stream)
                self.map_resolution = self.map_metadata["resolution"]
                self.map_path = self.map_metadata["image"]
                self.map_origin = self.map_metadata["origin"]

            except yaml.YAMLError as exc:
                print(exc)

    def compute_lookup_table(self, map_image_path: str) -> np.ndarray:
        """
        Compute the likelihood field lookup table
        :param map_image_path: path to the map image
        :return: likelihood field lookup table
        """

        # Read the map image
        map_image = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)

        # save where the grey pixels are
        self.grey_pixels = np.where(map_image == 205)

        # Threshold the map image to obtain binary representation (0 for obstacles, 255 for free space)
        _, binary_map = cv2.threshold(map_image, 206, 255, cv2.THRESH_BINARY)

        # Compute the distance transform: distance of free space to obstacles
        distance_transform = cv2.distanceTransform(binary_map, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        self.lookup_table = distance_transform * self.map_resolution

        return self.lookup_table

    def generate_likelihood_field_img(self, show_img: bool = False) -> None:
        """
        Generate the likelihood field image from the lookup table
        """

        # Normalize the distance transform to range [0, 1]
        lf_img = self.lookup_table / np.max(self.lookup_table)
        lf_img = np.max(lf_img) - lf_img
        lf_img = (lf_img * 255).astype(np.uint8)

        # add grey pixels to the lookup table
        lf_img[self.grey_pixels] = 205

        # Save the lookup table as an image
        cv2.imwrite("imgs/likelihood_field.png", lf_img)

        if show_img:
            cv2.imshow("Likelihood Field", lf_img)
            cv2.waitKey()
            cv2.destroyAllWindows()


    def get_likelihood_field(self, x: float, y: float) -> float:
        """
        Get the likelihood field value at a given position
        :param x: x-coordinate in the world frame
        :param y: y-coordinate in the world frame
        :return: likelihood field value at the given position
        """

        # Get the pixel coordinates from the world coordinates
        x_pixel = int((x - self.map_origin[0]) / self.map_resolution)
        y_pixel = int((y - self.map_origin[1]) / self.map_resolution)

        # print(x_pixel, y_pixel)

        # Return the likelihood field value at the pixel coordinates
        return self.lookup_table[y_pixel, x_pixel]


def main():
    lf = LikelihoodField()
    lf.compute_lookup_table(lf.map_path)
    lf.generate_likelihood_field_img(show_img=False)

    x = 200 * lf.map_resolution + lf.map_origin[0]
    y = 200 * lf.map_resolution + lf.map_origin[1]
    # print(f"Likelihood field value at ({x}, {y}): {lf.get_likelihood_field(x, y) / lf.map_resolution}")

    print("Likelihood field value at (0, 0):", lf.get_likelihood_field(0.981119133366672, 9.2) / lf.map_resolution)
    # print("Likelihood field value at (1,1):", lf.get_likelihood_field(1, 1) / lf.map_resolution)
# Example usage:
if __name__ == "__main__":

    main()