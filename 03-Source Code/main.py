from Preprocessing import *
from Segmentation import *

ShowSteps = 1
for filename in sorted(glob.glob('../02-Dataset/Training_Data/*.png')):
    # filename = "csep1220.png"  # The Rotated Image
    SegementedLines = []
    SegementedWordsPerLine = []  # Rows represent Line# and Col represent Words belong to # Line
    img = cv2.imread(filename)
    img = Preprocess(img, ShowSteps)
    SegementedLines = SegementedImageLines(img, ShowSteps)
    SegementedWordsPerLine = getWordImages(SegementedLines,ShowSteps)