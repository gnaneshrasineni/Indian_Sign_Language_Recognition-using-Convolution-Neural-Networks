import cv2
import time
import numpy as np
import os


def nothing(x):
    pass


image_x, image_y = 64, 64

def create_folder(folder_name):
    if not os.path.exists('./test/' + folder_name):
        os.makedirs('./test/' + folder_name)
        
       
def capture_images():
    #create_folder(str(ges_name))
    
    cam = cv2.VideoCapture(2)

    cv2.namedWindow("test_image_acquistion")

    img_counter = 0
    t_counter = 1
    #training_set_image_name = 1
    #test_set_image_name = 1
    listImage = [1,2,3,4,5]

    cv2.namedWindow("Trackbars")

    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 21, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 50, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 39, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    for loop in listImage:
        while True:

            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)

            l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")

            img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

            lower_blue = np.array([l_h, l_s, l_v])
            upper_blue = np.array([u_h, u_s, u_v])
            imcrop = img[102:298, 427:623]
            hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            result = cv2.bitwise_and(imcrop, imcrop, mask=mask)

            cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
            cv2.imshow("test", frame)
            cv2.imshow("mask", mask)
            cv2.imshow("result", result)

            if cv2.waitKey(1) == ord('c'):

                if t_counter <= 5:
                    img_name = "./test/{}.png".format(t_counter)
                    save_img = cv2.resize(mask, (image_x, image_y))
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    #training_set_image_name += 1

                t_counter += 1
                if t_counter == 6:
                    break
                img_counter += 1


            elif cv2.waitKey(1) == ord('b'):
                break

            if t_counter> 6:
                break


    cam.release()
    cv2.destroyAllWindows()
    
#ges_name = input("Enter gesture name: ")
capture_images()
