from commonfunctions import *


def SegementedImageLines(img, rShowSteps):
    # hist = cv2.reduce(thresh, 0, cv2.REDUCE_AVG).reshape(-1)
    #  plt.hist(thresh.ravel(), 256, [0, 255])
    #  plt.show()

    img = 255 - img
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    thresh = img
    # lines segmentation involves horizontal projection of the image rows to find the empty rows between rows that contain text a
    img_lines = np.copy(255 - img)
    i = 0  # iteratior on the rows of image(NxM)
    ListOfImageLines = []  # Contains the Segemented Lines Images
    while i < img.shape[0]:
        if np.sum(thresh[i, :]) != thresh.shape[1] * 255:
            First_Point = i - 1
            while np.sum(thresh[i, :]) != thresh.shape[1] * 255:
                i = i + 1
            LastPoint = i + 2  # for ي
            i = i + 2
            if abs(First_Point - LastPoint) < 10:
                continue  # not a line :( points
            partition = np.copy(thresh[First_Point:LastPoint, :])
            ListOfImageLines.append(partition)
            img_lines[i, :] = (np.ones(thresh.shape[1]))  # LINE END
        else:
            i = i + 1
        # Successfully Segmented the lines

    # For Printing Purposes & Illustration =>
    LineCount = 1
    if rShowSteps:
        for LineImage in ListOfImageLines:
            show_images([255 - LineImage], ["Line( " + str(LineCount) + ") :"])
            LineCount += 1
        show_images([255 - img, img_lines], ["Orignal Image ", "Image Withe Lines "])

    return ListOfImageLines


def skel(img):
    size = np.size(img)
    skell = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    for i in range(0, 2):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skell = cv2.bitwise_or(skell, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skell


def sortSecond(val):
    return val[0]


def getWordImages(ListofImageLines):
    for i in ListofImageLines:
        ret, thresh = cv2.threshold(i, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        kerneltext = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # dilation = cv2.erode(thresh, kerneltext, iterations=1)
        # dilation = cv2.dilate(dilation, kernel, iterations=1)
        dilation = 255 - thresh
        # dilation = cv2.dilate(dilation, np.array([1]), iterations=4)
        # dilation = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)

        io.imshow(dilation, cmap="binary")
        io.show()

        connectivity = 4
        output = cv2.connectedComponentsWithStats(dilation, connectivity, cv2.CV_32S)

        stats = output[2]

        stats = stats[stats[:, 0].argsort()]


        for j in range(0, output[2].shape[0]):
            if stats[j, cv2.CC_STAT_WIDTH] > 2 and stats[j, cv2.CC_STAT_AREA  ]>25:
                start = stats[j, cv2.CC_STAT_LEFT]
                end = start + stats[j, cv2.CC_STAT_WIDTH]
                print(stats[j])
                opening = cv2.morphologyEx(dilation[:, start:end], cv2.MORPH_OPEN, kernel2)

            # opening = cv2.erode(dilation[:, start:end], kernel2, iterations=1)

                io.imshow(dilation[:, start:end], cmap="binary")
                io.show()
