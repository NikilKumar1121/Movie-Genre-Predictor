from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle


def imagepredictor(file):

    pickle_in = open("obj.pkl","rb")
    classes  = pickle.load(pickle_in)
    loaded_model = load_model('model.h5')
    img = image.load_img(file,target_size=(350,350,3))
    img = image.img_to_array(img)
    img = img/255
    img = img.reshape(1,350,350,3)
    y_prob = loaded_model.predict(img)
    top_3_pred = np.argsort(y_prob[0])[:-4:-1]
    GENERES =[]
    for i in range(3):
        GENERES.append(classes[top_3_pred[i]])
    return GENERES
