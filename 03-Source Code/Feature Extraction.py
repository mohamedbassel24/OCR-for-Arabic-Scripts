from commonfunctions import *
from scipy import ndimage


def findLetterContourArea(img):
    # This function finds the contours of a given image and returns it in the variable contours.
    # This function will not work correctly unless you preprocess the image properly as indicated.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    framearea = cv2.contourArea(contours[0])
    # TODO: Find the contour area of the given image (img) (~1 line)
    maxArea = cv2.contourArea(contours[1])
    index=1
    for i in range(1,len(contours)):
        if(cv2.contourArea(contours[i])>maxArea):
            maxArea=cv2.contourArea(contours[i])
            index=i
    #area = cv2.contourArea(contours[1])
    return maxArea, contours[index],framearea

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

def Height_Width_Ratio(img):
    area, letter_contour,framearea = findLetterContourArea(img)
    whiteArea = framearea-area
    x, y, w, h = cv2.boundingRect(letter_contour)
    #print(x)
    #print(y)
    #print(h)
    #print(w)
    return h/w,   area/whiteArea


image=io.imread("test character letters/qaaff.png")
h_w,r=Height_Width_Ratio(image)

print(h_w)
print(r)
