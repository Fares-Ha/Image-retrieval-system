import numpy as np
from numpy.linalg import norm
import pickle
import io
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from PIL import Image

from flask import Flask, request

app = Flask(__name__)

print('0')
model = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False,
                 input_shape=(224, 224, 3), pooling='max')


def extract_features2(imge, model):
    input_shape = (224, 224, 3)
    img = imge.resize((input_shape[0], input_shape[1]), Image.ANTIALIAS)
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features


def get_ids_by_vectors(indices):
    vv = []
    for i in indices[0]:
        vv.append(id_feature_list[i][0])
    return vv


with open('id_feature_list', 'rb') as f:
    id_feature_list = pickle.load(f)

print('00')


@app.route('/get_sim', methods=['POST', 'GET'])
def get_sim():
    print('get_sim')
    im_file = request.files["image"]
    im_bytes = im_file.read()
    im = Image.open(io.BytesIO(im_bytes))

    feature_list = []
    for i in range(len(id_feature_list)):
        feature_list.append(id_feature_list[i][1])

    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute',
                                 metric='euclidean').fit(feature_list)

    ss = extract_features2(im, model)

    distances, indices = neighbors.kneighbors([ss])

    return json.dumps(get_ids_by_vectors(indices))


@app.route('/add', methods=['POST', 'GET'])
def add():
    print('add')
    files = request.files.getlist("image")
    print(files)
    idd = request.form["idjj"]
    print(idd)
    for im_file in files:
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        id_feature_list.append((int(idd), extract_features2(im, model)))

    with open('id_feature_list', 'wb') as f:
        pickle.dump(id_feature_list, f)

    return json.dumps('success')


@app.route('/delete', methods=['POST', 'GET'])
def delete():
    print('delete')
    idd = request.form["idjj"]
    print(idd)
    i = 0
    while True:
        if i == len(id_feature_list):
            break
        if id_feature_list[i][0] == idd:
            print(i)
            id_feature_list.pop(i)
            i = i - 1
        i = i + 1

    with open('id_feature_list', 'wb') as f:
        pickle.dump(id_feature_list, f)

    return json.dumps('success')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)