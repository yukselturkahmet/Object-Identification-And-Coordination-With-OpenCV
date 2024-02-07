import cv2
#Ahmet yükseltürk
#please add these things to your library: numphy, opencv-python,pillow
print("Hello! Welcome to ÇinÇin Restaurant")
name = input("Please write your name to create an order: ")
print("Welcome" + ' ' + name + ' ' + "here is the menu. You can only choose one from each category.")

print("Our menu icludes: Pizza, Pasta, Soup, Dessert")
print("Pizza options; car=Margarita: 50 TL, bicycle=BBQ chicken pizza: 60 TL,airplane=Supreme Pizza: 90 TL\nPasta options: bird=Chicken Alfredo: 70 TL, cat=Carbonara: 60 TL, dog=Bolognese: 60 TL\nSoup options: cup=Broccoli cheddar soup: 50 TL, fork=Tomato soup: 50 TL, knife=Chicken noodle soup: 60 TL\nDessert Options: banana=Magic cookie bars: 30 TL, orange=Brownies: 40 TL, apple=Cheesecake:70 TL")



# Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)


# Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)



# Initialize camera
cap = cv2.VideoCapture(0)


while True:
    # Get frames
    ret, frame = cap.read()


    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200,0,50), 3)

        class_name = classes[class_id]

        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, (200,0,50), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

