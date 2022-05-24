from nomralizeRGB import *
from circularMask import *
from arcDetect import *
from contorsHighlight import *



circularMaskRadius = 3

# Initialize background subtraction mask
backgroundSub = cv2.createBackgroundSubtractorMOG2()

# By default, detection function is applied and this is the first run of the software
applyDetection = True

firstRun = True

blockageCounter = 2
recordTime = 0
maxNoisePercentage = 50

flag = True

while(flag):

    # Read original frame.
    frame = cv2.imread('Image/RS-48@RS-49_RS-48_201911201533_02.jpg')

    # Normalize the frame (Reduce the noise).
    normalizedImage = normRGB(frame)

    # Applying background subtraction.
    bsImage = backgroundSub.apply(normalizedImage)

    # Convert to grayscale.
    grayImage = cv2.cvtColor(normalizedImage, cv2.COLOR_BGR2GRAY)

    # Apply edge detection filter to the image.
    cannyImage = cv2.Canny(grayImage, 200,20)

    # Thresholding the edges.
    ret, thresholdImage = cv2.threshold(cannyImage, 0, 255, cv2.THRESH_BINARY)

    # Apply distance transform to the thresholded image
    dat = cv2.distanceTransform(thresholdImage, cv2.DIST_L2 ,3)

    # Creating circle mask to the middle of the frame to block all noise.
    size = (frame.shape[1], frame.shape[0])
    
    # Positioning and sizing the circular mask
    maskRadius = size[1] // circularMaskRadius
    maskCenter = [int(size[0]/2), int(size[1]/2)]

    # Applying circular mask to frame
    mask = create_circular_mask(size[0], size[1], maskCenter, maskRadius)
    mask_area = np.pi*maskRadius**2
    maskedImage = bsImage.copy()

    # cv2.imshow("Image", maskedImage)
    # cv2.waitKey(0)

    maskedImage[~mask] = 0

    # Detect circles in images to block them from using black mask.
    # This is because sewage pipes usually are connected by rings, which are not defects.

    circles = cv2.HoughCircles(maskedImage, cv2.HOUGH_GRADIENT, 1, 20, param1=200, param2=20,
                               minRadius=int(maskRadius/2), maxRadius=int(maskRadius))

    # If detection is circle and apply detection is ON, block it by a circle mask.
    if circles is not None and applyDetection:
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:

            # Draw the outer circle
            cv2.circle(maskedImage, (i[0], i[1]), i[2], (0, 0, 0), 20)

            # Draw the maskCenter of the circle
            arcDenoisedImage = detectArcs(dat, maskedImage, 2, circles)
            contouredImage = highlightContours(arcDenoisedImage)

    # Show processed images.
    cv2.imshow('Processed Frame', frame)

    cv2.waitKey(0)

    flag = False

cv2.destroyAllWindows()

