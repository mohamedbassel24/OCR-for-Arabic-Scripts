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
