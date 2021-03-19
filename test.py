import cv2
img = cv2.imread("./1.jpg")
while True:
    cv2.imshow("out", img)
    ret = cv2.waitKey(0)
    print(ret)