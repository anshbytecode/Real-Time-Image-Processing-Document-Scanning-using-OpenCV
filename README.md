# Documentation-Scanner-Opencv-Python
It is the documentation scanner project of our Computer Vision work with Opencv. In this project, documents shown to the computer camera were scanned using Opencv artificial intelligence libraries. Below I tried to explain the details of the project step by step.


## Documentation-Sacnner-Opencv

First, we will download the libraries we will use.
```Python
import cv2
import numpy as np
import utlis

```

### Camera Options

```Python
webCamFeed = True
cap = cv2.VideoCapture(1)
cap.set(10, 160)
heightImg = 640
widthImg = 480
```
We add the necessary lines of code to start our webcam in the operations here.
If you do not want to operate with the webcam, you can also scan any document image into the program. (pathImage = "image name".
We have defined #camera screen widths above.
```Python
utlis.initializeTrackbars()
count = 0
```
```Python
while True:

    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))  
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8) 
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  .
    thres = utlis.valTrackbars()  
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1]) 
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

```
```Python

    imgContours = img.copy()  
    imgBigContour = img.copy()  
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) 
```
```Python

    biggest, maxArea = utlis.biggestContour(contours)
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest) 
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]) 
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) 

        
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

       
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])

    else:
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

   
    lables = [["Orijinal", "Gri", "Esik", "Contours"],
              ["En buyuk Contour", "Warp Perspektif", "Warp Gri", "Uyarlanabilir Esik"]]

    stackedImage = utlis.stackImages(imageArray, 0.75, lables)
    cv2.imshow("Program", stackedImage)
```
# Program Output

![myimage0](https://user-images.githubusercontent.com/67556543/182640541-1ebc36e9-c126-4195-9740-50f52b355301.jpg)

![myimage2](https://user-images.githubusercontent.com/67556543/182640564-73061e56-0853-407d-8981-de62a9e7f996.jpg)
