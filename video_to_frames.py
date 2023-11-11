## Program: reading frames of a video and storing it into pictures
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

################################### Parameter: input(video path)
file_name = './video1/output.mp4';

cap = cv2.VideoCapture(file_name) #Reading video

################################### Parameter: output(pictures path)
frame_path = 'C:/Users/NM/Documents/Python/Final_Project/Data1' #Addressdehi bayad motlagh bashe
os.chdir(frame_path) # change directory (cd) => in kar ro mikonim ta daghighan image ha tuye library output zakhire beshan.

################################### Parameter: frame_skip (chand ta chand ta frame ha ro skip kone)
frame_skip = 10

i = 0 # baraye ta'yeen name tasvire az in moteghayer estefade shode

while True:
    # Capture frame-by-frame
    ret, I = cap.read()

    for _ in range(frame_skip):
        if ret == False: # end of video (perhaps)
            break
        ret,chert = cap.read() # be tedad frame hayee ke mikhaym skip konim, alaki in moteghayer ha ro por mikonim

    if ret == False: # end of video (perhaps)
        break # age be akharin frame video residim, bayad kharej shim dige. chon frami namunde ke bekhaym bekhunim

    img_name = str(i) + ".jpg" #esme tasviri ke mikhaym zakhire konim ro mirizim tuye ye moteghayyer
    cv2.imwrite(img_name, I) #har image ro tuye 'output directory' zakhire mikonim. I=Image...
    i = i + 1 # esme har image yek adade. ba'd az zakhire sazi har image baraye zakhire sazi image ba'di un adad ro yedune ziad mikonim

cap.release()
# cv2.destroyAllWindows()
