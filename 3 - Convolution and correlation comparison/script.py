from copy import deepcopy
import cv2
import numpy as np

# Provide the image path here 
# img_path = r"pic.jpg"
img_path = r"3 - Convolution and correlation comparison/pic2.webp"

def get_kernel_for_convolution(kernel):
    outputKernel = []
    for i in range(len(kernel)):
        _originalRow = deepcopy(kernel[i])
        _originalRow.reverse()
        outputKernel.insert(0, _originalRow)
    return outputKernel

# flag=0 for grayscale
image = cv2.imread(img_path, 0)

(rows, cols) = image.shape  #if we print => (height=rows=1080, width=cols=1920)

correlational_kernel = [[1,2,3],[4,5,6],[7,8,9]]
correlation_kernel_sum = 0
for i in correlational_kernel:
    correlation_kernel_sum += sum(i)

convolution_kernel = get_kernel_for_convolution(correlational_kernel)
convolution_kernel_sum = 0
for i in correlational_kernel:
    convolution_kernel_sum += sum(i)

correlation_output_image = np.zeros((rows-2, cols-2), dtype=int)
convolution_output_image = np.zeros((rows-2, cols-2), dtype=int)

for r in range(0, rows-1):
    # print(r)
    if (r == 0 or r == rows - 1): 
        continue
    for c in range(0, cols):
        if (c == 0 or c == cols - 1):
            continue
        output_value_for_correlation = image[r-1][c-1] * correlational_kernel[0][0] + image[r][c-1] * correlational_kernel[0][1]  +image[r+1][c-1] * correlational_kernel[0][2] 
        output_value_for_correlation += image[r-1][c] * correlational_kernel[1][0] + image[r][c] * correlational_kernel[1][1] + image[r+1][c] * correlational_kernel[1][2]
        output_value_for_correlation += image[r-1][c+1] * correlational_kernel[2][0] + image[r][c+1] * correlational_kernel[2][1] + image[r+1][c+1] * correlational_kernel[2][2]

        correlation_output_image[r-1][c-1] = output_value_for_correlation/correlation_kernel_sum

        output_value_for_convolution = image[r-1][c-1] * convolution_kernel[0][0] + image[r][c-1] * convolution_kernel[0][1]  +image[r+1][c-1] * convolution_kernel[0][2] 
        output_value_for_convolution += image[r-1][c] * convolution_kernel[1][0] + image[r][c] * convolution_kernel[1][1] + image[r+1][c] * convolution_kernel[1][2]
        output_value_for_convolution += image[r-1][c+1] * convolution_kernel[2][0] + image[r][c+1] * convolution_kernel[2][1] + image[r+1][c+1] * convolution_kernel[2][2]

        convolution_output_image[r-1][c-1] = output_value_for_convolution/convolution_kernel_sum


cv2.imwrite('3 - Convolution and correlation comparison/correlation_output.jpg', correlation_output_image)
cv2.imwrite('3 - Convolution and correlation comparison/convolution_output.jpg', convolution_output_image)