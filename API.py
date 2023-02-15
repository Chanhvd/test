from flask import Flask, request
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

symbols =  "0123456789" # All symbols captcha can contain string.ascii_lowercase +
num_symbols = len(symbols)
new_model = tf.keras.models.load_model('model3.h5')

def predict_captcha(img):

    img = img / 255.0
    res = np.array(new_model.predict(img[np.newaxis, :, :, np.newaxis]))
    ans = np.reshape(res, (3, 10))
    l_ind = []
    probs = []
    for a in ans:
        l_ind.append(np.argmax(a))
       
    capt = ''
    for l in l_ind:
        capt += symbols[l]
    return capt

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'Image not found in request', 400
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    # resized_img = cv2.resize(img, (50, 200))
    try:
        captcha = predict_captcha(img)
    except:
        return ({'error': 'captcha model không hỗ trợ'}), 400
    return captcha
if __name__ == '__main__':
    app.run(debug=True)
