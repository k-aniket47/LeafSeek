from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)

MODEL_PATH ='model_inceptionV3.h5'
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)
   
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Plant is Alpinia Galanga (Rasna)"
    elif preds==1:
        preds="The Plant is Amaranthus Viridis (Arive-Dantu)"
    elif preds==2:
        preds="The Plant is Artocarpus Heterophyllus (Jackfruit)"
    elif preds==3:
        preds="The Plant is Azadirachta Indica (Neem)"
    elif preds==4:
        preds="The Plant is Basella Alba (Basale)"
    elif preds==5:
        preds="The Plant is Brassica Juncea (Indian Mustard)"
    elif preds==6:
        preds="The Plant is Carissa Carandas (Karanda)"
    elif preds==7:
        preds="The Plant is Citrus Limon (Lemon)"
    elif preds==8:
        preds="The Plant is Ficus Auriculata (Roxburgh fig)"
    elif preds==9:
        preds="The Plant is Ficus Religiosa (Peepal Tree)"
    elif preds==10:
        preds="The Plant is Hibiscus Rosa-sinensis"
    elif preds==11:
        preds="The Plant is Jasminum (Jasmine)"
    elif preds==12:
        preds="The Plant is Mangifera Indica (Mango)"
    elif preds==13:
        preds="The Plant is FMentha (Mint)"
    elif preds==14:
        preds="The Plant is Moringa Oleifera (Drumstick)"
    elif preds==15:
        preds="The Plant is Muntingia Calabura (Jamaica Cherry-Gasagase)"
    elif preds==16:
        preds="The Plant is Murraya Koenigii (Curry)"
    elif preds==17:
        preds="The Plant is Nerium Oleander (Oleander)"
    elif preds==18:
        preds="The Plant is Nyctanthes Arbor-tristis (Parijata)"
    elif preds==19:
        preds="The Plant is Ocimum Tenuiflorum (Tulsi)"
    elif preds==20:
        preds="The Plant is Piper Betle (Betel)"
    elif preds==21:
        preds="The Plant is Plectranthus Amboinicus (Mexican Mint)"
    elif preds==22:
        preds="The Plant is Pongamia Pinnata (Indian Beech)"
    elif preds==23:
        preds="The Plant is Psidium Guajava (Guava)"
    elif preds==24:
        preds="The Plant is Punica Granatum (Pomegranate)"
    elif preds==25:
        preds="The Plant is Santalum Album (Sandalwood)"
    elif preds==26:
        preds="The Plant is Syzygium Cumini (Jamun)"
    elif preds==27:
        preds="The Plant is Syzygium Jambos (Rose Apple)"
    elif preds==28:
        preds="The Plant is Tabernaemontana Divaricata (Crape Jasmine)"
    else:
        preds="The Plant is Trigonella Foenum-graecum (Fenugreek)"
    
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        result=preds
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
