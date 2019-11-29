from commonfunctions import *
from scipy import ndimage



def Count_connected_parts(img):    # this function returns the number of connected parts given the binary image of any letter
    labeled, nr_objects = ndimage.label(img < 100)  # 100 is the threshold but in case of binary image given (0,1) it will change
    #print(nr_objects)
    #print("Number of objects is {}".format(nr_objects))
    return nr_objects


def count_holes(img,num_connected_parts):    # count number of holes in each character
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(gray,-1,kernel)
    ret,thresh1 = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE   )
    #print("y= ",len(contours)-1-num_connected_parts)
    return len(contours)-1-num_connected_parts       # -1 is the contour of the image frame