from File_Accessing import *
from Preprocessing import *
from Segmentation import *
from classifiers import *
from NN_classifier import *

# Model Variables =>
alphabetic_Dict = {0: 'ا', 1: 'ب', 2: 'ت', 3: 'ث', 4: 'ج', 5: 'ح', 6: 'خ', 7: 'د', 8: 'ذ', 9: 'ر', 10: 'ز', 11: 'س',
                   12: 'ش', 13: 'ص', 14: 'ض', 15: 'ط', 16: 'ظ', 17: 'ع', 18: 'غ', 19: 'ف', 20: 'ق', 21: 'ك', 22: 'ل',
                   23: 'م', 24: 'ن', 25: 'ه', 26: 'و', 27: 'ي', 28: 'ؤ'}
classifier_list = list(alphabetic_Dict.keys())
alphabetic_list = list(alphabetic_Dict.values())
Model_Mode = 1  # 1 for TESTING 0 for TRAINING
Running_Time = []  # Representing the running time for each file
ShowSteps = 0  # Representing the steps of each block

if Model_Mode == 0:  # Training MODE
    for filename in sorted(glob.glob('../02-Dataset/Training_Data/*.png')):
        # filename = "csep1220.png"  # Test Image
        SegmentedLines = []  # Rows represent Line#
        SegmentedWordsPerLine = []  # Rows represent Line# and Col represent Words belong to # Line
        SegmentedCharacters = []  # Rows represent Characters
        img = cv2.imread(filename)  # Reading the Image
        TextImage = ParsingFile(filename[:len(filename) - 4])  # Removing .png from image path And Read the TextImage
        WordTextIndex = 0  # representing the #word in the text_file as we are moving sequential in the file
        Train_Set = []  # representing the data points needed to append in file
        Fault_Segment = 0  # representing # of wrong char segmentation
        # Pre processing
        start_time = time.time()
        img = Preprocess(img, ShowSteps)
        # Start Segmentation
        SegmentedLines = SegmentedImageLines(img, 0)
        SegmentedWordsPerLine = myVersion_GetWORDS(SegmentedLines, 0)

        Count_Word = 0
        for i in range(len(SegmentedWordsPerLine)):
            Count_Word += len(SegmentedWordsPerLine[i])
        print("Image Word#", Count_Word, "Text Word#:", len(TextImage))
        if Count_Word != len(TextImage):
            print("Error in file #:" + filename)
            continue
        # print("Length of Segmented: ", len(SegmentedLines))
        for SegmentedWords in SegmentedWordsPerLine:
            for Word in SegmentedWords:
                CharactersPerWord = []
                CharactersPerWord = getCharImages(Word, ShowSteps, WordTextIndex)
                TextImage[WordTextIndex] = Preprocesss_text(TextImage[WordTextIndex])
                if len(CharactersPerWord) != len(TextImage[WordTextIndex]):
                    WordTextIndex += 1
                    Fault_Segment += 1
                    continue
                for Char, Char_Text in zip(CharactersPerWord, TextImage[WordTextIndex]):
                    Char_Feature = Extracting_features(Char)
                    Labeled_Feature = [Char_Feature, classifier_list[alphabetic_list.index(Char_Text)]]
                    Train_Set.append(Labeled_Feature)

                WordTextIndex += 1
        end_time = time.time()
        print(WordTextIndex, len(TextImage))
        print("Appending to training set with accuracy of char segmentation =>",
              ((WordTextIndex - Fault_Segment) / WordTextIndex) * 100, " % Running Time:", end_time - start_time,
              "for File " + filename[-9:])
        Append_TraingSET(Train_Set)
else:
    RunningTimeOutput = ""
    Dir = "../test/"
    for filename in sorted(glob.glob(Dir + '*.png')):
        #    filename = "capr3.png"  # Test Image\
        FileNameImage = filename[len(Dir):]
        FileNameImage = FileNameImage[:len(FileNameImage) - 4]
        SegmentedLines = []  # Rows represent Line#
        SegmentedWordsPerLine = []  # Rows represent Line# and Col represent Words belong to # Line
        SegmentedCharacters = []  # Rows represent Characters
        img = cv2.imread(filename)  # Reading the Image
        WordTextIndex = 0  # representing the #word in the text_file as we are moving sequential in the file
        Output_Text = ""  # representing arabic text i will classify
        # READ MODEL
     #   X, Y = ReadModel("data_point")
        # Pre processing
        # SVM_Model = SVM_linear_training(X[0:30000, :], Y[0:30000])
        start_time = time.time()

        img = Preprocess(img, ShowSteps)


        # Start Segmentation
        SegmentedLines = SegmentedImageLines(img, ShowSteps)

        SegmentedWordsPerLine = myVersion_GetWORDS(SegmentedLines, ShowSteps)

        # myVersion_GetWORDS(SegmentedLines, ShowSteps)
        for SegmentedWords in SegmentedWordsPerLine:
            for Word in SegmentedWords:
                Word_Text = ""
                CharactersPerWord = []

                CharactersPerWord = getCharImages(Word, ShowSteps, WordTextIndex)

                WordTextIndex += 1
                for Char in CharactersPerWord:
                    Char_Feature = Extracting_features(Char)
                    # Using SVM Classifier
                    # myClassifier = SVM_Classifier(SVM_Model, [Char_Feature])
                    myClassifier = Predict_NN(Char_Feature)
                    myClassifier = myClassifier.item()

                    Char_Classified = ""
                    if myClassifier == 28:
                        Char_Classified = "لا"
                    else:
                        Char_Classified = alphabetic_Dict[myClassifier]
                    # show_images([Char], ["Predicted :" + Char_Classified])
                    Word_Text += Char_Classified
                Output_Text += " " + Word_Text
        #  Output_Text += "\n"

        Write_ClassifiedText(Output_Text, FileNameImage)
        end_time = time.time()
        RunningTimeOutput += str(end_time - start_time) + "\n"
    #print("Running Time :", end_time - start_time)
    Write_RunTime(RunningTimeOutput)
