from commonfunctions import *
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, skeletonize, thin
from scipy import stats
from Feature_Extraction import *
from scipy.ndimage import gaussian_filter


def SegmentedImageLines(img, rShowSteps):
    """" Given The Image get The text per line """
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
            img_lines[i, :] = 100 * (np.ones(thresh.shape[1]))  # LINE END
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


def getWordImages(ListofImageLines, rShowSteps):
    """" Given The Images Per Lines Get Words for each Line"""
    ListOfWordsPerLine = []
    for i in ListofImageLines:
        # ret, thresh = cv2.threshold(i, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilation = 255 - i
        if rShowSteps:
            io.imshow(dilation, cmap="binary")
            io.show()
        connectivity = 4
        output = cv2.connectedComponentsWithStats(dilation, connectivity, cv2.CV_32S)
        stats = output[2]
        stats = stats[stats[:, 0].argsort()]
        if rShowSteps:
            print(stats)

        stats = stats[(stats[:, cv2.CC_STAT_WIDTH] > 1) & (stats[:, cv2.CC_STAT_AREA] > 10)]
        j = 1
        ListOfWords = []
        while (True):
            if j + 1 > stats.shape[0]:
                break
            start = stats[j, cv2.CC_STAT_LEFT]
            end = start + stats[j, cv2.CC_STAT_WIDTH]
            while (j + 1 < stats.shape[0]) and (end + 4 >= stats[j + 1, cv2.CC_STAT_LEFT]):
                end = stats[j + 1, cv2.CC_STAT_LEFT] + stats[j + 1, cv2.CC_STAT_WIDTH]
                j += 1
                # print(stats[j, cv2.CC_STAT_LEFT],stats[j, cv2.CC_STAT_WIDTH],end)
            if rShowSteps:
                print(start, end, j)

            ListOfWords.insert(0, dilation[:, start:end])
            j += 1
        if rShowSteps:
            for X in ListOfWords:
                io.imshow(X, cmap="binary")
                io.show()
        ListOfWordsPerLine.append(ListOfWords)
    return ListOfWordsPerLine


def getCharImages(Word, ShowSteps, WordTextIndex):
    """" GET Char per Word """
    WordCount = 0
    Characters = []

    scale_percent = 200  # percent of original size
    width = int(Word.shape[1] * scale_percent / 100)
    height = int(Word.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    # Word = cv2.resize(Word, dim, interpolation=cv2.INTER_AREA)

    partition = np.copy(Word)
    # PreProcessing for the word ======================>
    img = np.copy(partition)
    partition = skeletonize(partition * 255)
    #  partition=thin(partition,max_iter=1)
    # partition=1-partition
    # partition=gaussian_filter(partition, sigma=0.2)
    partition[partition > 0] = 1
    # show_images([partition], ["wee"])
  #  if ShowSteps:
   #     show_images([partition, img], ["SubWord (" + str(WordCount + 1) + " )", "Smoothing"])
    # BASELINE DETECTION =>

    MAX = 0
    Base_INDEX = 0
    for i in range(partition.shape[0]):
        CurrMax = np.sum(partition[i])
        if CurrMax >= \
                MAX:
            MAX = CurrMax
            Base_INDEX = i
    # -------------------------------------------------------------------------------------
    # Max transitions DETECTION =>
    MaxTransition = 0
    MaxTransitionIndex = Base_INDEX
    for i in range(0, Base_INDEX, 1):  # loop on Each row
        CurrTransitionRow = 0
        flag = 0
        for j in range(partition.shape[1]):  # loop on coloumns for specific row
            if flag == 0 and partition[i, j] == 1:
                flag = 1
                CurrTransitionRow += 1
            elif flag == 1 and partition[i, j] == 0:
                flag = 0

        if CurrTransitionRow >= MaxTransition:
            MaxTransitionIndex = i
            MaxTransition = CurrTransitionRow
    # ----------------------------------------------------
    # ----- GET The mode of vertical projections which represents the line between chars
    m = stats.mode([sum(x) for x in zip(*partition)])
    MFV = m[0][0]
    # -----------------------------------------------------------------------------------
    # CutPoint Algorithm =>
    flag = 0
    ListOfCuts = []
    FirstRegion = True
    WordCount += 1
    # Background => zero Word =>1
    for i in range(partition.shape[1]):  # loop on the parition Col
        if partition[MaxTransitionIndex, i] == 0 and flag == 0:  # Check if its a background in the MTI After Char
            StartIndex = i  # Start of my Cut
            flag = 1  # Search for the First Char
        elif partition[MaxTransitionIndex, i] == 1 and flag == 1:  # Change from back ground to New Char
            EndIndex = i  # End of my Cut
            flag = 0
            MiddleIndex = int((StartIndex + EndIndex) / 2)  # average
            if StartIndex == 0 and abs(StartIndex - EndIndex) < 5:  # abs(StartIndex - EndIndex) < 2 or  # Skip the first cut between nothing and char
                continue
            #if FirstRegion:  # First region needs a special handling for char like ب ى ت
             #   if np.sum(partition[:, :StartIndex - 1]) > 5:
              #      FirstRegion = False

            if WordTextIndex == -1:  # For debuging search for a specific word
                print("This is a dummy condition for Debuging")

            SHPA = np.sum(
                partition[0:Base_INDEX - 1, :MiddleIndex])  # SUM OF HORIZONTAL PROJECTION Above BaseLine
            SHPB = np.sum(
                partition[Base_INDEX + 1:, :MiddleIndex])  # SUM OF HORIZONTAL PROJECTION Above BaseLine

            # for space seperation ?
            ThereIsGap = False
            for k in range(abs(StartIndex - EndIndex) + 1):
                CurrVP = np.sum(partition[:, StartIndex + k])
                if CurrVP == 0:
                    # print("Detect Gap in the Word ")
                    MiddleIndex = k + StartIndex
                    ThereIsGap = True
                    break
            if ThereIsGap:
                ListOfCuts.insert(0, MiddleIndex)
                if FirstRegion:
                    FirstRegion = False
                continue
            if FirstRegion:
                SHPA = np.sum(partition[0:Base_INDEX - 1,
                              0:EndIndex])  # SUM OF HORIZONTAL PROJECTION Above BaseLine
                SHPB = np.sum(partition[Base_INDEX:,
                              0:EndIndex])  # SUM OF HORIZONTAL PROJECTION Above BaseLine
                FirstRegion = False
                if SHPA == 0:  # for ya2
                    if Count_connected_parts(Word[:, :EndIndex]) == 3:
                        ListOfCuts.insert(0, EndIndex - 3)
                        continue
                if SHPA > SHPB and (MFV == np.sum(partition[:, MiddleIndex]) or MFV == np.sum(
                        partition[:, MiddleIndex - 1]) or MFV == np.sum(partition[:, MiddleIndex + 1])):
                    k = StartIndex
                    ConcaveFound = False
                    while np.sum(partition[0: MaxTransitionIndex, k]) != 0:
                        k += 1

                        if k >= EndIndex:
                            ConcaveFound = True
                            break
                    if ConcaveFound:  # false cut a word has a hole
                        continue

                    ListOfCuts.insert(0, MiddleIndex)
                    continue
                else:
                    continue

            # For Holes Detection
            k = StartIndex
            ConcaveFound = False
            while np.sum(partition[0: MaxTransitionIndex, k]) != 0:
                k += 1
                if k >= EndIndex:
                    ConcaveFound = True
                    break
            if ConcaveFound:  # false cut a word has a hole
                continue
            else:
                MiddleIndex = k  # adjst the middle index
            BaseLineDetection = np.sum(partition[Base_INDEX, StartIndex:EndIndex])
            if BaseLineDetection == 0 and SHPB >= SHPA:
                continue
            if MFV == np.sum(
                    partition[:, MiddleIndex]):  # VP[Middle]== Most Frequent Value veritucal Projection
                ListOfCuts.insert(0, MiddleIndex)
                continue
            ThereExistVP = False
            for k in range(abs(StartIndex - EndIndex)):
                if np.sum(partition[:, StartIndex + k]) <= MFV:
                    if ShowSteps:
                        print("Detect MFV")
                    MiddleIndex = k + StartIndex
                    ThereExistVP = True
                    break
            if ThereExistVP:
                ListOfCuts.insert(0, MiddleIndex)
                continue

            if SHPA > SHPB and MFV == np.sum(partition[:, MiddleIndex]):
                ListOfCuts.insert(0, MiddleIndex)
            else:
                continue
            ListOfCuts.insert(0, MiddleIndex)

    # Do Filteration here

    start = partition.shape[1]
    # Remove Cut for stroke =>
    for i in range(len(ListOfCuts) ):
        partition_Char = Word[:, ListOfCuts[i] + 1:start]
        start = ListOfCuts[i]
        Binary_Word = np.copy(Word)
        Binary_Word[Binary_Word > 0] = 1
        m = stats.mode([sum(x) for x in zip(*Binary_Word)])
        MFV_Before = m[0][0]  # skeleton

        if IsStroke(partition_Char, MFV_Before) and i+2 <len(ListOfCuts) :
            Old_Start = start
            partition_Char_After = Word[:, ListOfCuts[i + 1] + 1:start]
            start = ListOfCuts[i + 1]
            partition_Char_2ndAfter = Word[:, ListOfCuts[i + 2] + 1:start]
            start = ListOfCuts[i + 2]
            if IsStroke(partition_Char_After, MFV) or IsStroke(partition_Char_2ndAfter, MFV):
                ListOfCuts[i] = -1
                ListOfCuts[i + 1] = -1  # false cut
                i += 2
            else:
                start = Old_Start
        if IsDal(partition_Char):
            ListOfCuts[i - 1] = -1


            # i += 3

    StrokeList = []
    if ShowSteps:
        for Cut in ListOfCuts:
            if Cut == -1:
                continue
            img[:, Cut] = np.ones(img.shape[0]) * 150

    if ShowSteps:
        show_images([partition, img], ["SubWord (" + str(WordTextIndex) + " )", "Smoothing"])

    ListOfCuts.append(0)
    #      print(ListOfCuts)

    start = partition.shape[1]
    for Cut in ListOfCuts:
        if Cut == -1:
            continue
        partition_Char = Word[:, Cut + 1:start]
        start = Cut
        Characters.append(partition_Char)
    #  show_images([partition_Char], ["SubChar"])
    #  show_images([partition_Char], ["SubChar"])

    #if ShowSteps:
       # print(StrokeList)
       # show_images([partition, img], ["SubWord", "Smoothing"])
    return Characters


def IsStroke(Parition, MFV):
    # Parition = skeletonize(Parition * 255)
    Parition[Parition > 0] = 1
    # Parition=1-Parition
    Parition = Parition.astype('uint8')

    """" DETECT if the char is stroke"""
    # SINGLE COMPONENT =>

    scale_percent = 200  # percent of original size
    width = int(Parition.shape[1] * scale_percent / 100)
    height = int(Parition.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
#resized = cv2.resize(Parition, dim, interpolation=cv2.INTER_AREA)
    #     show_images([resized], ["We ?"])
    # cv2.re

    Count_Components = Count_connected_parts(Parition)
    if Count_Components > 1:
        return False

    Base_INDEX = 0
    MAX = 0
    for i in range(Parition.shape[0]):
        max = np.sum(Parition[i])
        if max >= MAX:
            MAX = max
            Base_INDEX = i
    SHPA = np.sum(Parition[0:Base_INDEX - 1])  # SUM OF HORIZONTAL PROJECTION Above BaseLine
    SHPB = np.sum(Parition[Base_INDEX + 1:])  # SUM OF HORIZONTAL PROJECTION Above BaseLine
    if SHPB > SHPA:
        return False
    area, letter_contour, framearea = findLetterContourArea(Parition)
    whiteArea = framearea - area
    if area ==-1:
        return False
    x, y, w, h = cv2.boundingRect(letter_contour)
    MaxHorizontalProjection = 0
    HorizontalList = []
    for row in range(np.shape(Parition)[0]):
        CurrProjection = np.sum(Parition[row])
        if CurrProjection > MaxHorizontalProjection:
            MaxHorizontalProjection = CurrProjection
            HorizontalList.append(MaxHorizontalProjection)
    SecondMVP = 0
    if len(HorizontalList) < 2:
        SecondMVP = HorizontalList[-1]
    else:
        SecondMVP = HorizontalList[-2]
    # sub from baseline ?
    if h - Base_INDEX > SecondMVP * 2:
        return False
   # print(h-Base_INDEX)
   # if h >5:
    #    return False
    m = stats.mode([sum(x) for x in zip(*Parition)])
    if m[0][0] != MFV:
    #    print(MFV, m[0][0])
        return False
    # NO HOLES =>
    if count_holes(Parition, Count_Components) > 0:
        return False
    #  print("Is Stroke ")
    return True


def StrokeDetection(char_img):
    # feature 1
    # feature 2
    # feature 3
    # feature 4
    return 1


def IsDal(char_img):
    """" This function to detect the stroke in dal """
    Count_White = np.sum(char_img == 1)
    if (Count_White < 3):
        #print("Is dal")
        return True

    return False
