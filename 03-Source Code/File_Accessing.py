import time
from commonfunctions import *


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
    return splitLine


def Append_TraingSET(Traing_Points):
    """ This function to append new training point to our data set"""
    f = open("data_point.txt", "a+")
    for Point in Traing_Points:
        f.write(str(Point[0][0]) + " " + str(Point[0][1]) + " " + str(Point[0][2]) + " " + str(Point[0][3]) + " " + str(
            Point[0][4]) + " " + str(Point[1]) + "\n")


def ReadModel(FileName):
    ListReadChar = []

    f = open(FileName + ".txt", 'r')
    if f.mode == "r":  # check if file is open
        FileContents = f.readlines()  # read file line by line
    else:
        return " "
    splitLine = np.zeros((np.shape(FileContents)[0] - 1, 6))
    for i in range(1, np.shape(FileContents)[0]):  # loop for the lines before last
        FileContents[i] = FileContents[i][0:len(FileContents[i]) - 1]
        Point = FileContents[i].split(' ')
        for j in range(6):

            if j == 5:
                splitLine[i - 1, j] = int(Point[j])
            else:
                splitLine[i - 1, j] = float(Point[j])

        # plitLine.append(FileContents[i].split(' '))
    # print(splitLine)
    return splitLine[:, 0:5], splitLine[:, 5]



