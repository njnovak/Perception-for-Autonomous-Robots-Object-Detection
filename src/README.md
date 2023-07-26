To run these files, matplotlib, opencv, and numpy are needed along with native python libraries.

Each file has been ran independently beforehand to confirm that it runs. Either run them from an IDE or type `$python3 FILE_NAME.py` in terminal, where `FILENAME` is the file you want to run.

I commented out any unnecessary `cv2.waitKey`calls. However, if intermediate images are desired for viewing while running, simply uncomment these `waitKey()` calls.

All results are saved as their own files and may be accessed within this folder as well. Ball detection via Hough Transforms' results are labeled `ballDetected.mov`. Rail detection has the lines detected named as `linesOnRails.jpg`, the warped image named as `warpedRails.jpg`, and the final plot of distance between the rails throughout the warped image named as `DistancePlot.jpg`. Hot-air balloon detection has it's results named as `DetectedBalloons.jpg`.

Any inquiries or other miscellaneous issues may be emailed to nnovak@umd.edu.
