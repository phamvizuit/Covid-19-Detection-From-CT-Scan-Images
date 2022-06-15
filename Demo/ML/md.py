import joblib
import cv2
def KNN(path):
    model1 = joblib.load("./models/KNN.joblib")
    img = cv2.imread(path)
    img = cv2.resize(img,(64,64))
    img = img.reshape(1,-1)
    if model1.predict(img) == 0:
        return "COVID"
    return "non-COVID"

