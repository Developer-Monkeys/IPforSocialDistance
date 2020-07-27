
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

#argparse argument diyerek input output gibi giris cikislarimizi kolaylastircaz dosya yollari belirliyoruz
#display dedigimiz ekranda goruntulemek icin bu sekilde daha duzenli oldugunu dusunuyorum

ap = argparse.ArgumentParser() # daha mantıklı değişkenler kullanılmalı
ap.add_argument("-i", "--input", type=str, default="sosyal.mp4", 
help="") 

# Proje büyüdükçe global değişkenlere ihtiyaç olcak
# mesela tkinterla girdi yaparak buraya atama yapabiliriz dışarıdan 
#bunu şöyle denedim aslında ama ne kadar mantıklı bilemiyorum.
###########|||||||||||||||||||#

#VIDEO_PATH = 'C:/Users/Beyz9/Desktop/sosyalmesafe/videos/video.mp4'
#OUTPUT_PATH = 'C:/Users/Beyz9/Desktop/sosyalmesafe/output/output.avi'
############################||||||||||||||||||#
ap.add_argument("-o", "--output", type=str, default="output.avi", 
help="")
ap.add_argument("-d", "--display", type=int, default=1, help="cikti cercevesi")
args = vars(ap.parse_args())

#labelspath dedigimiz ise yolov3 yapiisndaki coco names dosyasinda islenecek isimler yazili onu yukluyoruz
# coco names dosyasındaki classları ekliyor 
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#yolov3 algortimasinin cfg ve weights dosyalarini ekliyoruz
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
print("*****olusturuluyor*****")

#open cvnin dnn modulunu kullanarak yolo agimizi yuklemis olduk
net=cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#yolo dan cikti katmanlari topluyor islememiz icin gerekecek :D artık orda değil
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] # bu kullanım iğrenç ama hata değil :(
print("erisim saglandi")

# video akisimizi baslatiyoruz girdigimiz input degeri ile else 0 dedigimiz ise yol vermezsek 
# otomatik olarak video kamerasi acilacaktir
vs = cv2.VideoCapture(args["input"] if args["input"] else 0) # default 0


####################BURDA VIDEOCAPTURE İÇİNE DİREK VİDEOPATH YAZIP KULLANABİLİRİZ


#cikis videomuzu none olarak baslattik : neden
writer = None

# while icindeki grabbed frame gibi yapi open cv nin en temel olmazsa olmaz seyleri :D 
# cerceveyi bu sekilde isliyoruz. video uzerindeki kareler uzerinde dongu baslatiyoruz.
while True:
    (grabbed, frame) = vs.read()

    if not grabbed: # yani true vermek yerine bunu koyalım o zaman While şartına?
                     # YAPI BÖYLE
        break

    # video boyutu buyuk oldugu icin resize ile boyut belirlendi : benim pc yarım saat uğraştı :D
    frame = imutils.resize(frame, width=700)

    #detect_people diger py dosyamda detect islemi gerceklestigi icin o fonksiyonu alarak yolo 
    #nesne algilama sonuclarini elde ediyoruz. ve sadece person olani aldik
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

    # sosyal mesafeye uymayanlar icin liste tutacagiz. mesafe kontrolu iicn hazirlaniyoruz
    violate = set()

    # insanlari aldigimizda en az iki kisiyi tespit ettigimizi dusunuyoruz.
    if len(results) >= 2:
        #centroids ile oklid hesabi yap
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean") # değişken tanımlamayaydın 1 harf çok olmuş

        #matris hesabi yap
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                #iki kisi cok yakinsa onlari kontrol et  sosyal mesafeye uymayan kisi listesine ekleyecegiz.
                if D[i, j] < config.MIN_DISTANCE:
                    violate.add(i)
                    violate.add(j)
    
    for (i, (prob, bbox, centroid)) in enumerate(results): 
        # döngülerde i, j yerine mantıklı kelimeler kullanalım
        # bbox dedgimiz detect yaptigimiz py dosyamizda sinirlayici kutuyu ve centroid icin 
        # koordinatlari belirliyoruz
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid

        #yesil rengini belirledik
        color = (0, 255, 0)

        # uymuyorlarsa listeye at kirmizi yap dedik.
        if i in violate:
            color = (0, 0, 255)

        #rectangle ve circle ile cizdirdik.
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    # puttext ve text ilede uymayanlar kisilerin saysini gosterdik
    text = "Sosyal Mesafeyi Ihlal eden kisi sayisi : {}".format(len(violate))

    cv2.putText(frame, text, (10, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    if args["display"] > 0:
        # yazdiriyorsun : OKAY :D
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        #q ile goruntu uzerinden cikis yapmamizi sagliyoruz
        if key == ord("q"):
            break

    # burdaki output kontrolü yukardaki argparserda oluşturulcak değişkenle kontrol edilebilir
    # 3-4 yerde ayrı ayrı kullnılmış, kullanıcı bazlı veri alırken işlemeden önce işlerken vs
    if args["output"] != "" and writer is None: 
        # bu tekrarı önlesin diye sanırım none tanımlanmış ama gereksiz
        # bir de kontrollerde none null vs kullanmayalım kod patlayabilirim diye bağırır
        # HER YERDE BÖYLE YAPILMIS NAPAAMMM

        # bu komutlarla ise kayit olmasi icin yaziyoruz.
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25,(frame.shape[1], frame.shape[0]), True)


        ################## AYNI SEKILDE BURDA OUTPUTPATH

    if writer is not None:
        writer.write(frame)
