import os
import argparse
import cv2
from WordSegmentation import wordSegmentation, prepareImg


class DirPaths:
    "paths to data"
    fnInPath = ''
    fnMiddlePath = ''
    fnOutPath = ''


def main():
    """reads images from data/ and outputs the word-segmentation to out/"""

    print("")

    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-fi', help="<inPath> (example: 01_deslanted/) ../../<inPath> to read the handwriting images from.")
    parser.add_argument('-fm', help="<middlePath> (example: 0.png/) ../../<inPath><middlePath> AND ../../<outPath><middlePath>")
    parser.add_argument('-fo', help="<outPath> (example: 02_segmented/) ../../<outPath> to write the recognized words to.")
    args = parser.parse_args()

    if args.fi:
        DirPaths.fnInPath = args.fi
        # print(DirPaths.fnInPath)
    if args.fm:
        DirPaths.fnMiddlePath = args.fm
        # print(DirPaths.fnMiddlePath)
    if args.fo:
        DirPaths.fnOutPath = args.fo
        # print(DirPaths.fnOutPath)

    # read input images from 'in' directory
    inPath = '../../' + DirPaths.fnInPath + DirPaths.fnMiddlePath
    outPath = '../../' + DirPaths.fnOutPath + DirPaths.fnMiddlePath

    if not os.path.exists(inPath):
        print("No such path!")
        return(1)

    imgFiles = sorted(os.listdir(inPath))
    print("Files in directory : " + " ".join(imgFiles))

    for (i,f) in enumerate(imgFiles):
        print('Segmenting words of sample ' + inPath + '%s'%f)

        # If a directory pass it
        if os.path.isdir(inPath + '%s'%f) and DirPaths.fnMiddlePath == '':
            print('This is not a file, passing: %s'%f)
            continue

        # read image, prepare it by resizing it to fixed height and converting it to grayscale
        img = prepareImg(cv2.imread(inPath + '%s'%f), 50)

        # execute segmentation with given parameters
        # -kernelSize: size of filter kernel (odd integer)
        # -sigma: standard deviation of Gaussian function used for filter kernel
        # -theta: approximated width/height ratio of words, filter function is distorted by this factor
        # - minArea: ignore word candidates smaller than specified area
        res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
        # write output to 'out/inputFileName' directory
        if not os.path.exists(outPath + '%s'%f):
            print("Created directory %s"%f)
            os.mkdir(outPath + '%s'%f)

        # iterate over all segmented words
        print('Segmented into %d words'%len(res))
        for (j, w) in enumerate(res):
            print(inPath + '%s'%f + " => " + outPath + '%s/%d.png'%(f, j))
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            cv2.imwrite(outPath + '%s/%d.png'%(f, j), wordImg) # save word
            cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image

        # output summary image with bounding boxes around words
        cv2.imwrite(outPath + '%s/summary.png'%f, img)
        print(inPath + '%s'%f + " => " + outPath + '%s/summary.png'%f + "\n")


if __name__ == '__main__':
    main()
