#################################USE This Code##############
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import ipdb
from skimage import io as sio
from skimage.transform import resize
from skimage.feature import hog
from skimage.color import rgb2gray
#from train import detect
import subprocess
from imutils.contours import sort_contours
import joblib
import time
from urllib.request import urlopen
import sys
from pathlib import Path
from datetime import datetime, timedelta
import RPi.GPIO as GPIO
from picamera import PiCamera
import json
import requests
import smtplib
import socket
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import datetime
from time import sleep
import shutil
from collections import deque
import csv
import pdb
import config_WM


def access_csv(device_id, column):
    li=[]
    with open('configs.csv') as csvfile:
        content = csv.reader(csvfile)
        header = next(content)
        for row in content:
            if row[0] == device_id:
                myList = row
        
    head = header.index(column)
    try:
        return json.loads(myList[head])
    except:
        return myList[head]

#data = access_csv('Dummy', 'Ireading')

'''
1. URI-THe path in OM2M
2. Headers--auth key,data format inforamtion,
3. actual request--post you will get a response which is http response ---201=Success
'''
def create_data_cin( uri_cnt, value, cin_labels="", fmt_ex="json"):
    """
        Method description:
        Creates a data content instance(data_CIN) in the OneM2M framework/tree
        under the specified DATA CON
        Parameters:
        uri_cnt : [str] URI for the parent DATA CON
        fmt_ex : [str] payload format (json/XML)
    """

    headers = {
        'X-M2M-Origin': '{}:{}'.format(
            "devtest",
            "devtest"
        ),
        'Content-type': 'application/{};ty=4; charset=utf-8'.format(fmt_ex)
    }

    payload = {
        "m2m:cin": {
            "con": "{}".format(value),
            # "con": (
            #     json.dumps(value)
            #     if fmt_ex == 'json'
            #     else "{}".format(value)
            # ),
            "lbl": cin_labels,
            "cnf": "text"
        }
    }


    try:
        response = requests.post(uri_cnt, json=payload, headers=headers
                                 )
    except TypeError:
        response = requests.post(uri_cnt, data=json.dumps(payload),
                                 headers=headers)
    cin = None
    success = False
    if response.ok:
        cin = json.loads(response.content)['m2m:cin']['rn']
        success = True

    # _log.debug('Return code : {}'.format(response.status_code))
    # _log.debug('Return Content : {}'.format(response.text))
    return success, response.status_code, cin

'''
all the present logic here

'''
relay_pin = 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(relay_pin, GPIO.OUT)
# camera = PiCamera()
WRITE_API = access_csv(config_WM.device_id, "write_api")               #config_WM.write_api		# Write API of Himalaya_parking
BASE_URL = "https://api.thingspeak.com/update?api_key={}".format(WRITE_API)
# Meter coordinates, starting from top-left in clockwise manner
pts_source = np.float32(access_csv(config_WM.device_id, "pts_source"))	#[[391,311], [1747,319], [1750, 715], [389,685]])
width, height = 650, 215
pts_dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
MIN_CONTOUR_AREA = 1500
RESIZED_IMAGE_WIDTH = 45
RESIZED_IMAGE_HEIGHT = 90
CROP_COORD = 540
memory_path = "/home"

def checkInternetSocket(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        print(ex)
        return False
    
def get_time():
    # Get time stamp
    ct = datetime.datetime.now()
    time_stamp = (str(ct.year) + '-' + str(ct.month) + '-' + str(ct.day) +' '+
    str(ct.hour) + ':' + str(ct.minute) + ':' + str(ct.second))
    return time_stamp

def wait():
    # calculate the delay to the start of next minute
    next_minute =(datetime.datetime.now() + timedelta(seconds=25))
    delay = (next_minute - datetime.datetime.now()).seconds
    time.sleep(delay)
    
def cam(save_path):
    camera = PiCamera()
    camera.start_preview()
    time.sleep(2)
    camera.capture(save_path)
    print('Captured')
    camera.stop_preview()
    camera.close()
    
device_id = config_WM.device_id		#"PH-03"
time_stamp = get_time()
mail_content = time_stamp
#mail addresses and password
'''
sender_address = 'bawankarnilesh1420@gmail.com'
sender_pass = 'Nilesh@14'
receiver_address = 'twosixeight969@gmail.com'
'''
sender_address = access_csv(config_WM.device_id, "sender_address")
sender_pass = access_csv(config_WM.device_id, "sender_pass")
receiver_address = access_csv(config_WM.device_id, "receiver_address")

subject_text =  device_id
attach_file_name = time_stamp + '.jpg'
save_path = '/home/pi/Desktop/images/' + attach_file_name
    
def get_sorted_contour(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.GaussianBlur(img_gray, (9, 9), 0)
    img_gray = cv2.medianBlur(img_gray, 15)
    thresh = cv2.adaptiveThreshold(img_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    33, 5)
    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.dilate(thresh,kernel,iterations = 5)
    thresh = cv2.erode(thresh,kernel,iterations = 2)
    #plt.figure()
    #plt.imshow(thresh)
    #plt.title("Threshold Image")
    #plt.show()
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours, bbox = sort_contours(contours)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= MIN_CONTOUR_AREA]
    for contour in contours: # for each contour
        [intX, intY, intW, intH] = cv2.boundingRect(contour) # get and break out bounding rect
        # draw rectangle around each contour
        cv2.rectangle(img, # draw rectangle on original training image
        (intX, intY), # upper left corner
        (intX+intW,intY+intH), # lower right corner
        (0, 0, 255), # red
        3) # thickness
    #plt.figure()
    #plt.imshow(img)
    #plt.title("Detected Contours")
    #plt.show()
    return contours


def func(save_path, Filename):
    global cons
    img = sio.imread(save_path) 
    #img = sio.imread("/home/pi/Desktop/images/img2021-08-05-22-52-56.jpg") 
    #plt.imshow(img)
    #plt.show()
    matrix = cv2.getPerspectiveTransform(pts_source, pts_dst)
    img_meter = cv2.warpPerspective(img, matrix, (width, height))
    img_meter = img_meter[:, :CROP_COORD]
    #img_meter = img	# If you need to detect digits without coordinates, then comment above line and uncomment current line.
    #plt.imshow(img_meter)
    #plt.title("Extracted Meter")
    #plt.show()
    
    contours = get_sorted_contour(img_meter.copy())
    model = joblib.load('rf_rasp_classifier.sav') # 
            
    result = ''
    for contour in contours:
        [intX, intY, intW, intH] = cv2.boundingRect(contour)
        imgROI = img_meter[intY:intY + intH, intX:intX + intW]
        img = cv2.cvtColor(imgROI, cv2.COLOR_BGR2GRAY)
        img = resize(img, (RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH))
        img_feat = hog(img, orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2))
        digit_detected = model.predict(img_feat.reshape(1, -1))
        result += str(digit_detected[0])
    
    if(len(contours)==7):
        result = int(result) / 10
    elif(len(contours)==6):
        result = int(result)
    else:
        print('skipped')
        #pass
        return None

          
    print("detected value:", result)
    time =  Filename[-1][3:-4]
    time = datetime.datetime.strptime(time, '%Y-%m-%d-%H-%M-%S')
    stored_value.append(result)
    i=0
#   pdb.set_trace()
    str_current = str(stored_value[-1])
    str_prev = str(stored_value[-2])
    print(len(str_prev))
    k=5
    if(len(str_current)!=8):
        if(len(str_current)==7):
            str_current=('0'+str_current)
        elif(len(str_current)==6):
            str_current=('00'+str_current)
        elif(len(str_current)==5):
            str_current=('000'+str_current)
        elif(len(str_current)==4):
            str_current=('0000'+str_current)
            
    if(len(str_prev)!=8):
        if(len(str_prev)==7):
            str_prev=('0'+str_prev)
        elif(len(str_prev)==6):
            str_prev=('00'+str_prev)
        elif(len(str_prev)==5):
            str_prev=('000'+str_prev)
        elif(len(str_prev)==4):
            str_prev=('0000'+str_prev)

    while((stored_value[-2]-stored_value[-1]>=10 or stored_value[-2]-stored_value[-1]<=-10 ) and (k >= 1)):
        current_value = (str_current[i])
        prev_value = (str_prev[i])
        current_value=int(current_value)
        prev_value=int(prev_value)
        print(current_value)
        print(prev_value)
        if(current_value - prev_value != 0):
            current_value =  prev_value- current_value
            stored_value[-1] = stored_value[-1] + (current_value * 10**k)
            print(stored_value[-1])
        i+=1
        k-=1
    print(stored_value[-2])
    print(stored_value[-1])
    print("dif=", stored_value[-2] - stored_value[-1])
    if cons >=0:
        if stored_value[-2] - stored_value[-1] >=0 :
            stored_value[-1]=stored_value[-2]
            print("Lesser")
            f_rate.append(0)
        elif  stored_value[-1] > stored_value[-2]:
            difference = stored_value[-1] - stored_value[-2]
            F1 = Filename[-1][3:-4]
            F2 = Filename[-2][3:-4]
            f1 = datetime.datetime.strptime(F1, '%Y-%m-%d-%H-%M-%S')
            f2 = datetime.datetime.strptime(F2, '%Y-%m-%d-%H-%M-%S')
            time_difference = (f1 - f2)
            p_sec = (time_difference.seconds) / 60
            r = round(difference/p_sec, 3)
            f_rate.append(r)

#   pdb.set_trace()
    print(stored_value[-1])
    print(stored_value[-2])
    print(f_rate)
    print(stored_value[-2] - stored_value[-1])    
    
    result_file = open(access_csv(config_WM.device_id, "vol_file"),'a')	
    result_file.writelines([str(time)  + ' ' + str(stored_value[-1]) + '\n'])
    result_file.close()
    
    print(stored_value[-2]- stored_value[-1])
    print(stored_value)
   
    
    result_file = open(access_csv(config_WM.device_id, "rate_file"), 'a')
    result_file.writelines([str(time)  + ' ' + str(f_rate[-1]) + '\n'])
    result_file.close()
    file_stored_value = str(stored_value[-1])
    #write
    print("Wrote on a file")
    f = open('Variable.txt','w')
    f.write(file_stored_value)
    f.close()

#write google-sheets, thingspeak or om2m code here   
    try:
        requests.get('https://script.google.com/macros/s/' + access_csv(config_WM.device_id, "gsheets") + '/exec?timestamp=%s&total_flow=%s&rate=%s'%(str(Filename[-1][3:-4]),str(stored_value[-1]),str(f_rate[-1])))
    except:
        print("Not send to google sheets")   
    cons+= 1
    #print(cons)
    


def send_email(sender_address,sender_pass,receiver_address,subject_text,save_path, Filename):
    #Setup the MIME
    message = MIMEMultipart()
    message['Subject'] = subject_text
    message['From'] = sender_address
    message['To'] = receiver_address

     #The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))
    attach_file = open(save_path, 'rb') # Open the file as binary mode
    data = attach_file.read()
    payload = MIMEBase('application', "octet-stream")
    payload.set_payload(data)
    encoders.encode_base64(payload) #encode the attachment
    #add payload header with filename
    payload.add_header("Content-Disposition", 'attachment', filename=Filename[-1])
    message.attach(payload)
 
    #Create SMTP session for sending the mail
    try:
        session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        session.starttls() #enable security
        session.login(sender_address, sender_pass) #login with mail_id and password
        text = message.as_string()
        session.sendmail(sender_address, receiver_address, text)
        session.quit()
        #os.remove(attach_file_name)
    except:
        print ("Error: unable to send email")

            
#     xvalue = []
#     plt.rcParams.update({'font.size': 18})
#     plt.plot(time_plot_new , stored_value, linewidth=2)
#     plt.grid(True)
#     plt.xlabel("Time ", fontsize= 25)
#     plt.ylabel("Kilo Liters", fontsize=25)
#     plt.show()
#     plt.rcParams.update({'font.size': 16})
#     plt.plot(time_plot, stored_value, linewidth=1) 
#     plt.grid(True)
#     plt.xlabel("Time ", fontsize=22)
#     plt.ylabel("Kilo Liters", fontsize=22)
#     # plt.title("Water volume", fontsize=30)
#     plt.show()
#     time_plot_new = time_plot_new[:-1]
#     time_plot = time_plot[:-1]
#     plt.rcParams.update({'font.size': 18})
#     plt.plot(time_plot_new,f_rate, linewidth=2)
#     plt.grid(True)
#     plt.xlabel("Time ",fontsize=22)
#     plt.ylabel("Kilolitre/minute", fontsize=22)
#     # plt.title("Rate of water flow", fontsize=30)
#     plt.show()
#     plt.rcParams.update({'font.size': 16})
#     plt.plot(time_plot, f_rate, linewidth=1)
#     plt.grid(True)
#     plt.xlabel("Time ", fontsize=22)
#     plt.ylabel("Kilolitre/minute", fontsize=22)
#     plt.show()
  

def main():
    try:
        while True:
        # set LED high
            t1 = int(time.time())
            Filename.append(str(datetime.datetime.now().strftime("img%Y-%m-%d-%H-%M-%S") + ".jpg"))
            print ("start")
            subprocess.call("/home/pi/Desktop/waterspcrc/"+ access_csv(config_WM.device_id, "fileD")+"/run_cmd_bash.sh")
            os.system("sudo /etc/init.d/ntp stop")
            try:
                os.system("sudo ntpdate " + access_csv(config_WM.device_id, 'time'))
            except:
                pass
            os.system("sudo /etc/init.d/ntp start")
            print ("end")
            print("Setting high - LED ON")
            GPIO.output(relay_pin, GPIO.HIGH)
            save_path = '/home/pi/Desktop/images/' + str(Filename[-1])
            cam(save_path)
            time.sleep(3)
            # set LED low
            print("Setting low - LED OFF")
            GPIO.output(relay_pin, GPIO.LOW)
            
            #check = checkInternetSocket(host="8.8.8.8", port=53, timeout=3)
            #if(check==True):
            #   send_email(sender_address,sender_pass,receiver_address,subject_text,save_path,Filename) 
            #else:
            #    print("mail not sent")
            
            daterec = []
            func(save_path, Filename)
            os.remove(save_path)
            t2 = int(time.time())
            # wait()
            delay = 60 - (t2-t1)
            if delay < 0:
                delay = 0

            time.sleep(delay)
    except KeyboardInterrupt:
        GPIO.cleanup()
    finally:
        print("executed successfully")

if __name__ == '__main__':
    stored_value = deque(5*[0], 5) # creating list 
    cons = 0
    f_rate = deque(5*[0], 5)
    Filename = deque(5*[0], 5)
    print("read")
    f = open('Variable.txt','r')
    reading_file = f.read()
    f.close()
    reading_file=float(reading_file)
    stored_value.append(reading_file)	#33875.3
    Filename.append(str(datetime.datetime.now().strftime("img%Y-%m-%d-%H-%M-%S") + ".jpg"))
    main()
    





