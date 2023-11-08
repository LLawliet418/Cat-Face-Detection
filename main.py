import cv2

# Initalize the camera
cap = cv2.VideoCapture(0)

# load the datasets
cat_face_cascade = cv2.CascadeClassifier(
    'assets/haarcascade_frontalcatface.xml')

while True:

    ret, img = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    cat_faces = cat_face_cascade.detectMultiScale(gray, 1.2, 5)

    for (top, right, bottom, left) in cat_faces:
        cv2.rectangle(img, (top, right), (top+bottom,
                      right+left), (252, 3, 78), 2)
        cv2.putText(img, 'Cat Face', (top, right-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252, 3, 78), 2)

    cv2.imshow('Cat Faces', img)

    if cv2.waitKey(30) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
