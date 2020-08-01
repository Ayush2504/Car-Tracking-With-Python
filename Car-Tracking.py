import cv2

# Our Image
#img_file ="car.jpg"
video = cv2.VideoCapture('videoplayback.mp4')


# Our Pre Trained car classifier
classifier_file = 'cars.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

while True:
    # Read the current Frame
    (read_successful, frame) =video.read()

    if read_successful:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars
    cars = car_tracker.detectMultiScale(grayscale_frame)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    

    cv2.imshow("Car Detector", frame)
    key=cv2.waitKey(1)

    if key==81 or key==113:
        break
    
"""
#create openCv image
img = cv2.imread(img_file)

#convert to grayscale
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(black_n_white)

#Draw rectangles around the cars
for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

# Display the image with car spotted
cv2.imshow("Car Detector", img)
cv2.waitKey()
"""
print("Code completed")