from Preprocessing import *
from Segmentation import *
from File_Accessing import *

# TODO: Create Dictionary for each Letter corresponding class# DONE
# TODO:Choose a best classier
# TODO: handle char segmentation and word accuracy
# TODO: impalement a logic for a testing
# TODO:Convert each  la to X in text image DONE
# TODO: # of segemented words != text word (bongo)

# Model Variables =>
alphabetic_Dict = {0: 'ا', 1: 'ب', 2: 'ت', 3: 'ث', 4: 'ج', 5: 'ح', 6: 'خ', 7: 'د', 8: 'ذ', 9: 'ر', 10: 'ز', 11: 'س',
                   12: 'ش', 13: 'ص', 14: 'ض', 15: 'ط', 16: 'ظ', 17: 'ع', 18: 'غ', 19: 'ف', 20: 'ق', 21: 'ك', 22: 'ل',
                   23: 'م', 24: 'ن', 26: 'ه', 27: 'و', 28: 'ي', 29: 'ؤ'}
classifier_list = list(alphabetic_Dict.keys())
alphabetic_list = list(alphabetic_Dict.values())
Operation_Mode = ["Training-Mode", "Test-Mode"]
Model_Mode = "Training-Mode"
Running_Time = []  # Representing the running time for each file
ShowSteps = 0  # Representing the steps of each block
# TRAINING DATA =>
# if Model_Mode == Operation_Mode[0]:
for filename in sorted(glob.glob('../02-Dataset/Training_Data/*.png')):
    # filename = "csep1220.png"  # Test Image
    SegmentedLines = []  # Rows represent Line#
    SegmentedWordsPerLine = []  # Rows represent Line# and Col represent Words belong to # Line
    SegmentedCharacters = []  # Rows represent Characters
    img = cv2.imread(filename)  # Reading the Image
    TextImage = ParsingFile(filename[:len(filename) - 4])  # Removing .png from image path And Read the TextImage
    WordTextIndex = 0  # representing the #word in the text_file as we are moving sequential in the file
    Train_Set = []  # representing the data points needed to append in file
    Fault_Segement = 0  # representing # of wrong char segmentation
    # Pre processing
    start_time = time.time()
    img = Preprocess(img, ShowSteps)
    # Start Segmentation
    SegmentedLines = SegmentedImageLines(img, ShowSteps)
    SegmentedWordsPerLine = getWordImages(SegmentedLines, ShowSteps)
    for SegmentedWords in SegmentedWordsPerLine:
        for Word in SegmentedWords:
            CharactersPerWord = []
            CharactersPerWord = getCharImages(Word, ShowSteps, WordTextIndex)
            TextImage[WordTextIndex] = Preprocesss_text(TextImage[WordTextIndex])
            if len(CharactersPerWord) != len(TextImage[WordTextIndex]):
                # print("Wrong segmentation :( for " + TextImage[WordTextIndex])
                #  show_images([Word], [str(WordTextIndex)])
                WordTextIndex += 1
                Fault_Segement += 1
                continue
            for Char, Char_Text in zip(CharactersPerWord, TextImage[WordTextIndex]):
                Char_Feature = Extracting_features(Char)
                Labeled_Feature = [Char_Feature, classifier_list[alphabetic_list.index(Char_Text)]]
                Train_Set.append(Labeled_Feature)
            WordTextIndex += 1
    end_time = time.time()
    print("Appending to training set with accuracy of char segmentation =>",
          ((WordTextIndex - Fault_Segement) / WordTextIndex) * 100, " % Running Time:", end_time - start_time,
          "for File " + filename[-9:])
    Append_TraingSET(Train_Set)
