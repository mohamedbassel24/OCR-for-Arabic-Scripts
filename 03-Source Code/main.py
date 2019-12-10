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
    # print(FileContents)
    splitLine = FileContents[0].split(' ')
    print(splitLine)
    # print(splitLine[1][1])
    return splitLine[1:]


# TRAINING DATA =>
for filename in sorted(glob.glob('../02-Dataset/Training_Data/*.png')):
    filename = "csep1220.png"  # The Rotated Image
    SegementedLines = []  # Rows represent Line#
    SegementedWordsPerLine = []  # Rows r epresent Line# and Col represent Words belong to # Line
    SegmentedCharacters = []  # Rows represent Characters
    img = cv2.imread(filename)  # Reading the Image
    TextImage = ParsingFile(filename[:len(filename) - 4])  # Removing .png from image path And Read the TextImage
    WordTextIndex = 0

    # Start Segementation
    img = Preprocess(img, ShowSteps)
    SegementedLines = SegementedImageLines(img, ShowSteps)
    SegementedWordsPerLine = getWordImages(SegementedLines, ShowSteps)
    for SegementedWords in SegementedWordsPerLine:
        for Word in SegementedWords:
            CharactersPerWord = []
            CharactersPerWord = getCharImages(Word, 0)
            # Char_Feature=Extracting_features()

            for Char, Char_Text in zip(CharactersPerWord, TextImage[WordTextIndex]):
                show_images([Char], ["Segemented Character for " + Char_Text])
                Char = Char.astype('uint8')
                print(np.shape(Char))
                print(Extracting_features((Char)))
            WordTextIndex += 1
