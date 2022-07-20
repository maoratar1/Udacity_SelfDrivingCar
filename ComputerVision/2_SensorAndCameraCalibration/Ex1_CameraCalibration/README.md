## Correcting For Distortion

![Distorted and Undistorted Images](orig-and-undist.png)

Here, you'll get a chance to try camera calibration and distortion correction for yourself!  

There are two main steps to this process: use chessboard images to obtain image points and object points, 
and then use the OpenCV functions `cv2.calibrateCamera()` and `cv2.undistort()` to compute the calibration and undistortion.  
