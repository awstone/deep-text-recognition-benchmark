import cv2


def binarize(img):
    ret, bin_img = cv2.threshold(img, 166, 255, cv2.THRESH_BINARY)
    return bin_img

def resize(img):
    return cv2.resize(img, (100, 32))

img = cv2.imread('./a01-000u-00.png')
#bin_im = binarize(img)
bin_im = resize(img)
cv2.imshow('fig', bin_im)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('binim.png', bin_im)
