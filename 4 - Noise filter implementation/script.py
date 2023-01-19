from typing import List
import cv2
import numpy as np
import math

def applyMeanFiltering(image, kernelSize):
    
    (rows, cols) = image.shape  # (height=rows, width=cols)
    correlation_kernel_sum = kernelSize * kernelSize # addition of all 'ones' in the kernel

    correlational_kernel = []
    for i in range(kernelSize):
        correlational_kernel.append([1 for k in range(kernelSize)])

    output_img_size = (rows - kernelSize + 1, cols - kernelSize + 1)
    correlation_output_image = np.zeros((rows, cols), dtype=int)

    a = math.floor(kernelSize/2)
    removed_row_or_col_starting_positions = [i for i in range(a)]
    
    removed_row_ending_positions = [rows - i - 1 for i in range(a)]
    removed_col_ending_positions = [cols - i - 1 for i in range(a)]

    iterating_list_for_kernel = [0]
    for i in removed_row_or_col_starting_positions:
        iterating_list_for_kernel.insert(len(iterating_list_for_kernel), i+1)
        iterating_list_for_kernel.insert(0,-1*(i+1))

    for r in range(0, rows):
        if (r in removed_row_or_col_starting_positions or r in removed_row_ending_positions): 
            continue
        for c in range(0, cols):
            if (c in removed_row_or_col_starting_positions or c in removed_col_ending_positions):
                continue
            
            output_value_for_correlation = 0
            for k1 in iterating_list_for_kernel:
                for k2 in iterating_list_for_kernel:
                    output_value_for_correlation += int(image[r+k1][c+k2])

            correlation_output_image[r][c] = output_value_for_correlation/correlation_kernel_sum
    correlation_output_image = correlation_output_image[a-1:len(correlation_output_image)]
    return correlation_output_image

# returns the median of the given array of numbers
def findMedian(numbers: List):

    #NOTE: the built-in sorting algorithm of Python uses a special version of merge sort, called Timsort, which runs in  nlog2n  time.
    numbers.sort()

    if(len(numbers) % 2 == 0): # even length array
        lowerMiddle = numbers[int(len(numbers) / 2) - 1]
        upperMiddle = numbers[int(len(numbers) / 2)]
        return (lowerMiddle + upperMiddle) / 2
    return numbers[int((len(numbers) - 1) / 2)]

def applyMedianFiltering(image, kernelSize):
    
    (rows, cols) = image.shape  # (height=rows, width=cols)
    correlation_kernel_sum = kernelSize * kernelSize # addition of all 'ones' in the kernel

    output_image = np.zeros((rows, cols), dtype=int)

    a = math.floor(kernelSize/2)
    removed_row_or_col_starting_positions = [i for i in range(a)]

    removed_row_ending_positions = [rows - i - 1 for i in range(a)]
    removed_col_ending_positions = [cols - i - 1 for i in range(a)]

    iterating_list_for_kernel = [0]
    for i in removed_row_or_col_starting_positions:
        iterating_list_for_kernel.insert(len(iterating_list_for_kernel), i+1)
        iterating_list_for_kernel.insert(0,-1*(i+1))

    for r in range(0, rows):
        if (r in removed_row_or_col_starting_positions or r in removed_row_ending_positions): 
            continue
        for c in range(0, cols):
            if (c in removed_row_or_col_starting_positions or c in removed_col_ending_positions):
                continue
            
            kernel_value_list = []
            for k1 in iterating_list_for_kernel:
                for k2 in iterating_list_for_kernel:
                    kernel_value_list.append(int(image[r+k1][c+k2]))
            output_image[r][c] = findMedian(kernel_value_list)

    return output_image


def applyThresholdAveraging(image, kernelSize, T):
    
    (rows, cols) = image.shape  # (height=rows, width=cols)
    correlation_kernel_sum = kernelSize * kernelSize # addition of all 'ones' in the kernel

    correlational_kernel = []
    for i in range(kernelSize):
        correlational_kernel.append([1 for k in range(kernelSize)])

    output_img_size = (rows - kernelSize + 1, cols - kernelSize + 1)
    correlation_output_image = np.zeros((rows, cols), dtype=int)

    a = math.floor(kernelSize/2)
    removed_row_or_col_starting_positions = [i for i in range(a)]

    removed_row_ending_positions = [rows - i - 1 for i in range(a)]
    removed_col_ending_positions = [cols - i - 1 for i in range(a)]

    iterating_list_for_kernel = [0]
    for i in removed_row_or_col_starting_positions:
        iterating_list_for_kernel.insert(len(iterating_list_for_kernel), i+1)
        iterating_list_for_kernel.insert(0,-1*(i+1))

    for r in range(0, rows):
        if (r in removed_row_or_col_starting_positions or r in removed_row_ending_positions): 
            continue
        for c in range(0, cols):
            if (c in removed_row_or_col_starting_positions or c in removed_col_ending_positions):
                continue
            
            output_value_for_correlation = 0
            for k1 in iterating_list_for_kernel:
                for k2 in iterating_list_for_kernel:
                    output_value_for_correlation += int(image[r+k1][c+k2])
            
            if (abs((output_value_for_correlation/correlation_kernel_sum) - image[r][c]) < T):
                correlation_output_image[r][c] = output_value_for_correlation/correlation_kernel_sum
            else:
                correlation_output_image[r][c] = image[r][c]
    return correlation_output_image

def applyKClosetAveraging(image, kernelSize, K):

    if (K > kernelSize * kernelSize):
        print("K value should be smaller or equal to the square of kernel size (number of columns or rows)")
        return False

    (rows, cols) = image.shape  # (height=rows, width=cols)
    correlational_kernel = []
    for i in range(kernelSize):
        correlational_kernel.append([1 for k in range(kernelSize)])

    correlation_output_image = np.zeros((rows, cols), dtype=int)

    a = math.floor(kernelSize/2)
    removed_row_or_col_starting_positions = [i for i in range(a)]
    
    removed_row_ending_positions = [rows - i - 1 for i in range(a)]
    removed_col_ending_positions = [cols - i - 1 for i in range(a)]

    iterating_list_for_kernel = [0]
    for i in removed_row_or_col_starting_positions:
        iterating_list_for_kernel.insert(len(iterating_list_for_kernel), i+1)
        iterating_list_for_kernel.insert(0,-1*(i+1))

    for r in range(0, rows):
        if (r in removed_row_or_col_starting_positions or r in removed_row_ending_positions): 
            continue
        for c in range(0, cols):
            if (c in removed_row_or_col_starting_positions or c in removed_col_ending_positions):
                continue
            
            kernel_value_and_difference_list = []
            for k1 in iterating_list_for_kernel:
                for k2 in iterating_list_for_kernel:
                    kernel_value_and_difference_list.append((image[r+k1][c+k2], abs(int(image[r+k1][c+k2]) - int(image[r][c])))) 

            sorted_list_by_difference = list(sorted(kernel_value_and_difference_list, key=lambda item: item[1]))
            list_to_be_averaged = [y[0] for y in sorted_list_by_difference]
            list_to_be_averaged = list_to_be_averaged[0:K]
            correlation_output_image[r][c] = sum(list_to_be_averaged)/K

    return correlation_output_image

# Provide the image path here 
img_path = r"4 - Noise filter implementation/pic22.png"

# flag=0 for grayscale
image = cv2.imread(img_path, 0)

# kernelSize should an odd  value (1, 3, 5, 7, ...) for better results

output_img = applyMeanFiltering(image, 7)
# output_img = applyMedianFiltering(image, 5)
# output_img = applyThresholdAveraging(image, 5, 210)
# output_img = applyKClosetAveraging(image, 5, 8)

cv2.imwrite('4 - Noise filter implementation/out.png', output_img)
