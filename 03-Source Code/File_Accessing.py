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
            Point[0][4]) + " " + str(Point[0][5]) + " " + str(Point[0][6]) + " " + str(Point[0][7]) + " " +
                str(Point[0][8]) + " " + str(Point[0][9]) + " " + str(Point[0][10]) + " " + str(
            Point[0][11]) + " " + str(
            Point[0][12]) + " " + str(Point[0][13]) + " " + str(Point[0][14]) + " " + str(Point[0][15]) + " " + str(
            Point[1]) + "\n")


def ReadModel(FileName):
    ListReadChar = []

    f = open(FileName + ".txt", 'r')
    if f.mode == "r":  # check if file is open
        FileContents = f.readlines()  # read file line by line
    else:
        return " "
    splitLine = np.zeros((np.shape(FileContents)[0] - 1, 17))
    for i in range(1, np.shape(FileContents)[0]):  # loop for the lines before last
        FileContents[i] = FileContents[i][0:len(FileContents[i]) - 1]
        Point = FileContents[i].split(' ')
        for j in range(17):

            if j == 16:
                splitLine[i - 1, j] = int(Point[j])
            else:
                splitLine[i - 1, j] = float(Point[j])

        # plitLine.append(FileContents[i].split(' '))
    # print(splitLine)
    return splitLine[:, 0:16], splitLine[:, 16]


def Write_ClassifiedText(Text,FileName):  # TODO: take file name
    """ This function to append new training point to our data set"""
    Dir="../output/text/"+FileName
    f = open(Dir+".txt", "w", encoding='utf-8')
    f.write(Text)

def Write_RunTime(Text):  # TODO: take file name
    """ This function to append new training point to our data set"""
    Dir="../output/runtime"
    f = open(Dir+".txt", "w", encoding='utf-8')
    f.write(Text)

