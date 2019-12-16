from Preprocessing import *
from Segmentation import *

ShowSteps = 0


def ParsingFile(FileName):
    ListReadChar = []
    f = open(FileName + ".txt", 'r', encoding='utf-8')
    if f.mode == "r":  # check if file is open
        FileContents = f.readlines()  # read file line by line
    else:
        return " "
    for i in range(np.shape(FileContents)[0]):  # loop for the lines before last
        FileContents[i] = FileContents[i][0:len(FileContents[i]) - 1]
    splitLine = FileContents[0].split(' ')
    return splitLine[1:]


def Append_TraingSET(Traing_Points):
    """ This function to append new training point to our dataset"""
    f = open("data_point.txt", "a+")

    for Point in Traing_Points:
        f.write(str(Point[0][0]) + " " + str(Point[0][1]) + " " + str(Point[0][2]) + " " + str(Point[0][3]) + " " + str(
            Point[0][4]) + " " + str(Point[1]) + "\n")


# TRAINING DATA =>
for filename in sorted(glob.glob('../02-Dataset/Training_Data/*.png')):
    filename = "csep1220.png"  # The Rotated Image
    SegementedLines = []  # Rows represent Line#
    SegementedWordsPerLine = []  # Rows r epresent Line# and Col represent Words belong to # Line
    SegmentedCharacters = []  # Rows represent Characters
    img = cv2.imread(filename)  # Reading the Image
    TextImage = ParsingFile(filename[:len(filename) - 4])  # Removing .png from image path And Read the TextImage
    WordTextIndex = 0
    Traing_Set = []
    # Start Segementation
    img = Preprocess(img, ShowSteps)
    SegementedLines = SegementedImageLines(img, ShowSteps)
    SegementedWordsPerLine = getWordImages(SegementedLines, ShowSteps)
    for SegementedWords in SegementedWordsPerLine:
        for Word in SegementedWords:
            CharactersPerWord = []
            CharactersPerWord = getCharImages(Word, 0)
            #            Char_Feature = Extracting_features()

            for Char, Char_Text in zip(CharactersPerWord, TextImage[WordTextIndex]):
                Char_Feature = Extracting_features(Char)
                print(Char_Feature)
                Labeled_Feature = [Char_Feature, Char_Text]
                print(Labeled_Feature)
                show_images([Char], ["Segemented Character for " + Char_Text])
                g = input("Do you want to Append ?y/n")
                if g == "y":
                    Traing_Set.append(Labeled_Feature)
                    print(Traing_Set)
            print("Appending =>")
            Append_TraingSET(Traing_Set)
            WordTextIndex += 1
