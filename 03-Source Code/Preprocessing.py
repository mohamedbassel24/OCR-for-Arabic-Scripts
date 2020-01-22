from commonfunctions import *


def Preprocess(img, showSteps):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]  # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(rotated, 155, 255, cv2.THRESH_BINARY)

    GlobalThresh = threshold_otsu(rotated)-100
    ThreshImage = np.copy(rotated)
    ThreshImage[rotated >= GlobalThresh] = 0
    ThreshImage[rotated < GlobalThresh] = 1
    if showSteps:
        show_images([255 - img, ThreshImage], ["Orignal Image ", "After Threholding And rotation"])
    return 255 - threshold
def Preprocesss_text(Text):
    """this function is converting لا into x"""
    if "لا" in Text:
        Text=Text.replace('لا', 'ؤ')
      #  Text=Text[::-1]
    return Text
#Str__="لا الله الا لله"
#print(Str__)
#Str__=Preprocesss_text(Str__)
#print(Preprocesss_text(Str__))