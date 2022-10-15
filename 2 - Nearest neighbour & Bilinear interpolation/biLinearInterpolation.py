import numpy as np
import cv2

# Provide the image path here 
img_path = r"2 - Nearest neighbour & Bilinear interpolation/pic2.jpg"

# specify the RGB or grayscal here in 2nd parameter (1 for RGB, 0 for grayscale)
image = cv2.imread(img_path, 1)

if(len(image.shape) == 3):
    height = image.shape[0]
    width = image.shape[1]
else: height, width = image.shape

# Give the desired dimensions here
# NOTE: 1920*1080 image has => width = 1920, height = 1080 
desired_height, desired_width = 320, 481

newImage = np.zeros((desired_height, desired_width, 3), dtype=int)

# Scaling factors
rows_scale_factor = desired_height/height
cols_scale_factor = desired_width/width

# Iterate through every pixel
for r in range(desired_height):
    for c in range(desired_width):

        # Mapped coordinates of the original image
        ori_col = c/cols_scale_factor
        ori_row = r/rows_scale_factor

        # Column and row values of the four neighbor points of the mapped point in original image
        col1 = min(int(np.floor(ori_col)), width-1)
        row1 = min(int(np.floor(ori_row)), height-1)
        col2 = min(int(np.ceil(ori_col)), width-1)
        row2 = min(int(np.ceil(ori_row)), height-1)

        # rgb values for the four neighbor points
        P11 = (image[row1, col1])
        P12 = (image[row2, col1])
        P21 = (image[row1, col2])
        P22 = (image[row2, col2])

        # Bilinear interpolation
        x = (ori_col - col1)
        y = (ori_row - row1)
        new_values = (1-x)*(1-y)*P11 + x*(1-y)*P12 + (1-x)*y*P21 + x*y*P22

        newImage[r, c] = new_values

# Saving the image
cv2.imwrite('2 - Nearest neighbour & Bilinear interpolation/scaled-bilinear.jpg', newImage)