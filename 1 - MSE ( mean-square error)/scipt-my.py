import cv2

#Computes MSE for the given original and procesed image and returns the MSE value.
# returns -1 if the two images are not compatible in shape
def grayscale_MSE(original_img, processed_img):  
    if (original_img.shape != processed_img.shape):
        print("The two pictures are not compatible with the shape")
        return -1
    
    isRGB = False
    if (len(original_img.shape) == 3):
        isRGB = True
        [rows, cols, dim] = original_img.shape
    else:
        [rows, cols] = original_img.shape

    # print(original_img[1, 2])
    result = 0
    for i in range (rows):
        for j in range (cols):
            if isRGB:
                for k in range (3):
                    result += (int(original_img[i, j][k]) - int(processed_img[i, j][k]))**2
            else: 
                result += (int(original_img[i, j]) - int(processed_img[i, j]))**2
    return result / (rows * cols)


original_img_path = r"1 - MSE ( mean-square error)/plane.jpg"    # specify the path for the original image
processed_img_path = r"1 - MSE ( mean-square error)/plane-edited.jpg"    # specify the path for the processed image

original_img_gs = cv2.imread(original_img_path, 0) # flag=0 for grayscale
processed_img_gs = cv2.imread(processed_img_path, 0) # flag=0 for grayscale

original_img_rgb = cv2.imread(original_img_path, 1) # flag=1 for rgb
processed_img_rgb = cv2.imread(processed_img_path, 1) # flag=1 for rgb

print(grayscale_MSE(original_img_gs, processed_img_gs))

