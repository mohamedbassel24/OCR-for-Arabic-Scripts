from commonfunctions import *


def SegementedImageLines(img):
    img = 255 - img
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # hist = cv2.reduce(thresh, 0, cv2.REDUCE_AVG).reshape(-1)
    #  plt.hist(thresh.ravel(), 256, [0, 255])
    #  plt.show()
    # lines segmentation involves horizontal projection of the image rows to find the empty rows between rows that contain text a

    SegemenetedRows = np.zeros((800, thresh.shape[1]))
    index = 0  # for the whole image with lines
    i = 0  # iteratior on the rows of image(NxM)
    ListOfImageLines = []  # Contains the Segemented Lines Images
    while i < img.shape[0]:
        if np.sum(thresh[i, :, 0]) != thresh.shape[1] * 255:
            First_Point = i - 1
            while np.sum(thresh[i, :, 0]) != thresh.shape[1] * 255:
                SegemenetedRows[index] = (thresh[i, :, 0])
                index = index + 1
                i = i + 1
            LastPoint = i + 2  # for ي
            i = i + 2
            if abs(First_Point - LastPoint) < 6:
                continue  # not a line :( points
            partition = np.copy(thresh[First_Point:LastPoint, :, 0])
            ListOfImageLines.append(partition)
            SegemenetedRows[index] = (np.ones(thresh.shape[1]))  # LINE END
            index = index + 1
        else:
            i = i + 1

    for LineImage in ListOfImageLines:
        io.imshow(LineImage, cmap="binary")  # 0~255 np.zeros((2, 1))
        io.show()
    ImageWithLine = SegemenetedRows[0:index, :]
    io.imshow(ImageWithLine, cmap="binary")  # 0~255 np.zeros((2, 1))
    io.show()
    return ListOfImageLines
