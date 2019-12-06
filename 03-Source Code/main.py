from Preprocessing import *
from Segmentation import *

ShowSteps = 0
for filename in sorted(glob.glob('../02-Dataset/Training_Data/*.png')):
    #filename = "csep1220.png"  # The Rotated Image
    SegementedLines = [] # Rows represent Line#
    SegementedWordsPerLine = []  # Rows represent Line# and Col represent Words belong to # Line
    SegmentedCharacters=[] # Rows represent Characters
    img = cv2.imread(filename)
    img = Preprocess(img, ShowSteps)
    SegementedLines = SegementedImageLines(img, ShowSteps)
    SegementedWordsPerLine = getWordImages(SegementedLines,ShowSteps)
    getCharImages(SegementedWordsPerLine,1)

