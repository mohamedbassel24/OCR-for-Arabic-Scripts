from Preprocessing import *
import sys

for filename in sorted(glob.glob('../02-Dataset/Training_Data/*.png')):
    #filename = "csep1220.png"  # The Rotated Image
    img = cv2.imread(filename)
    img = Preprocess(img)

    img = 255 - img
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    hist = cv2.reduce(thresh, 0, cv2.REDUCE_AVG).reshape(-1)
    plt.hist(thresh.ravel(), 256, [0, 255])
    plt.show()
    th = 0
    # lines segmentation involves horizontal projection of the image rows to find the empty rows between rows that contain text a

    SegemenetedRows = np.zeros((800, thresh.shape[1]))
    index = 0
    i = 0
    ListOfImageLines = []
    while i < thresh.shape[0]:

        if np.sum(thresh[i, :, 0]) != thresh.shape[1] * 255:
            First_Point = i-1
            while (np.sum(thresh[i, :, 0]) != thresh.shape[1] * 255):
                SegemenetedRows[index] = (thresh[i, :, 0])
                index = index + 1
                i = i + 1
            LastPoint = i+2 # for ي
            i=i+2
            if abs(First_Point-LastPoint) <6:
                continue # not a line :( points
            partition = np.copy(img[First_Point:LastPoint, :, 0])
            ListOfImageLines.append(partition)
            SegemenetedRows[index] = (np.ones(thresh.shape[1]))  # LINE ENDEND
            index = index + 1
        else:
            i = i + 1

    for LineImage in ListOfImageLines:

        io.imshow(LineImage, cmap="binary")  # 0~255 np.zeros((2, 1))
        io.show()
    # IndexsOfLines = y for y in range(thresh.shape[0]) if np.sum(thresh[y, :, 0]) != thresh.shape[1] * 255
    print("Dim: ", np.shape(SegemenetedRows))
    io.imshow(SegemenetedRows, cmap="binary")  # 0~255 np.zeros((2, 1))
    io.show()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow(filename, img)
    cv2.waitKey(0)
