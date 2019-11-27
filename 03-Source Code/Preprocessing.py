from commonfunctions import *


def Preprocess(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.blur(img, (3, 3))  # Blur the image to remove the noise (apply a Gaussian low pass filter 3x3)
    # _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)  #Convert the grayscale image to a binary image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]  # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        # otherwise, just take the inverse of the angle to make
        # it positive
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


for filename in sorted(glob.glob('../02-Dataset/Training_Data/*.png')):
    filename = "csep1220.png" # The Rotated Image
    img = cv2.imread(filename)
    img = Preprocess(img)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow(filename, img)
    cv2.waitKey(0)