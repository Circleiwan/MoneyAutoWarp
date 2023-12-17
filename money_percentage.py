import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
	pass
kernel = np.ones((5,5),np.uint8)

'''cv2.namedWindow("tracker")
cv2.createTrackbar("R", "tracker", 0, 255, nothing)
cv2.createTrackbar("G", "tracker", 0, 255, nothing)
cv2.createTrackbar("B", "tracker", 0, 255, nothing)'''

img = cv2.imread("uang_seratus.jpg")
img_copy = img.copy()
img_box = img.copy()
img_dot = img.copy()
#img_contour = cv2.imread("uang_seratus.jpg")
img_warped = img.copy()
img_with_line_no_warp= img.copy()
img_with_line_warp= img.copy()

while True:	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.blur(gray, (5,5))

	'''
	R = cv2.getTrackbarPos("R", "tracker")
	G = cv2.getTrackbarPos("G", "tracker")
	B = cv2.getTrackbarPos("B", "tracker")
	'''

	ret,th = cv2.threshold(blur,250,255,cv2.THRESH_BINARY)

	Mask = cv2.bitwise_not(th)
	opening = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	erosion = cv2.erode(closing, kernel, iterations = 2)

	contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(img_copy, contours, -1, (0,255,0), 3)
	cnt = contours[0]

	rect = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	print(box[1][0])

	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(img_box,(x,y),(x+w,y+h),(0,0,255),3)

	cv2.circle(img_dot, (x,y), 4, (255,0,0), -1)
	cv2.circle(img_dot, (x+w,y), 4, (255,0,0), -1)
	cv2.circle(img_dot, (x,y+h), 4, (255,0,0), -1)
	cv2.circle(img_dot, (x+w,y+h), 4, (255,0,0), -1)

	pts1 = np.float32([box[1],box[2], box[0], box[3]])
	pts2 = np.float32([[x,y-10],[x+w,y-10],[x,y+h],[x+w,y+h]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	dst = cv2.warpPerspective(img_warped, M, (299,145))

	#cv2.line(img_with_line_no_warp, box[1], box[1][0]+300, (0,0,255), 2)
	#cv2.line(img_with_line_warp, (x,y-10), (x,y-10)+100, (0,0,255), 2)

	cv2.imshow("Before Warped", img_with_line_no_warp)
	cv2.imshow("After Warped", img_with_line_warp)

	xywh = [x, w, y, h]
	warp_parameter = []
	for i in xywh:
		print(i)

	'''gray = np.float32(gray)
	dst = cv2.cornerHarris(gray, 2,3,0.04)
	dst = cv2.dilate(dst,None)
	img[dst>0.001*dst.max()]=[0,0,255]'''
	'''
	cv2.imshow("original", img)
	cv2.imshow("blur", blur)	
	cv2.imshow("threshold", th)
	cv2.imshow('Mask', closing)
	cv2.imshow('Erosion', erosion)
	'''
	images = [img, blur, th, closing, erosion, img_copy, img_box, img_dot, dst]
	titles = ["Original", "Blurred", "Thresholding", "Masking", "Morphological Transform", "Result", "Bounding Box", "Warp Point", "Warped/Result"]

	for i in range(9):
	    plt.subplot(3,3,i+1),plt.imshow(images[i], "gray")
	    plt.title(titles[i])
	    plt.xticks([]),plt.yticks([])
	
	plt.show()

	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()