from Preprocessing import *
from Segmentation import *

ShowSteps = 1
for filename in sorted(glob.glob('../02-Dataset/Training_Data/*.png')):
    # filename = "csep1220.png"  # The Rotated Image
    SegementedLines = []
    SegementedWords = []
    img = cv2.imread(filename)
    img = Preprocess(img,ShowSteps)
    SegementedLines = SegementedImageLines(img,ShowSteps)
    getWordImages(SegementedLines)
