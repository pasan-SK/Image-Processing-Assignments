import numpy as np
import cv2

def nearestneighbor(src_img, desired_height, desired_width):
    if isRGB:
        newImage = np.zeros((desired_height, desired_width, 3), dtype=int)
    else:
        newImage = np.zeros((desired_height, desired_width), dtype=int)

    # Scaling factors
    rows_scale_factor = desired_height/height
    cols_scale_factor = desired_width/width

    # Iterate through every pixel
    for r in range(desired_height):
        for c in range(desired_width):

            # nearest neighbor method
            nearest_c = int(np.round(c/cols_scale_factor))
            nearest_r = int(np.round(r/rows_scale_factor))

            if isRGB:
                newImage[r, c, 0] = image[nearest_r, nearest_c, 0]
                newImage[r, c, 1] = image[nearest_r, nearest_c, 1]
                newImage[r, c, 2] = image[nearest_r, nearest_c, 2]
            else: 
                newImage[r, c] = image[nearest_r, nearest_c]
    return newImage

# Provide the image path here 
img_path = r"2 - Nearest neighbour & Bilinear interpolation/pic2.jpg"

# specify the RGB or grayscal here in 2nd parameter (1 for RGB, 0 for grayscale)
image = cv2.imread(img_path, 0)

isRGB = False
if(len(image.shape) == 3): 
    isRGB = True 
    height = image.shape[0]
    width = image.shape[1]

else: height, width = image.shape

# Give the desired dimensions here
# NOTE: 1920*1080 => width (cols) = 1920, height(rows) = 1080 
desired_height, desired_width = 320, 480

newImage = nearestneighbor(image, desired_height, desired_width)

# Saving the image
cv2.imwrite('2 - Nearest neighbour & Bilinear interpolation/scaled-nearest-neighbor.jpg', newImage)