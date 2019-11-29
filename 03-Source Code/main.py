from Preprocessing import *
from Segmentation import *



for filename in sorted(glob.glob('../02-Dataset/Training_Data/*.png')):
    #filename = "csep1220.png"  # The Rotated Image
    img = cv2.imread(filename)
    img = Preprocess(img)
    SegementedLines=[]
    SegementedLines=SegementedImageLines(img)
    getWordImages(SegementedLines)
