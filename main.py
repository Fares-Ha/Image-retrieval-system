import numpy as np
from numpy.linalg import norm
import pickle
import io
import json

from sklearn.neighbors import NearestNeighbors
from PIL import Image

import torch
import torchvision
import torchvision.models as models

from flask import Flask, request


app = Flask(__name__)

torch.set_num_threads(1)

# Load the pretrained model
model = models.resnet50(pretrained=True)

# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

# Set model to evaluation mode
model.eval()

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_vector(image):
    # Create a PyTorch tensor with the transformed image
    t_img = transforms(image)
    # Create a vector of zeros that will hold our feature vector
    # The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(2048)

    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())                 # <-- flatten

    # Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # Run the model on our transformed image
    with torch.no_grad():                               # <-- no_grad context
        model(t_img.unsqueeze(0))                       # <-- unsqueeze
    # Detach our copy function from the layer
    h.remove()
    # Return the feature vector
    my_embedding = my_embedding.detach().numpy()
    my_embedding = my_embedding / norm(my_embedding)
    return my_embedding


def get_ids_by_vectors(indices):
    with open('/home/fareshm/mysite/id_feature_list', 'rb') as f:
        id_feature_list = pickle.load(f)
    vv = []
    for i in indices[0]:
        vv.append(id_feature_list[i][0])
    return vv


with open('/home/fareshm/mysite/id_feature_list', 'rb') as f:
    id_feature_list = pickle.load(f)


@app.route('/get_sim', methods=['POST', 'GET'])
def get_sim():
    with open('/home/fareshm/mysite/id_feature_list', 'rb') as f:
        id_feature_list = pickle.load(f)
        
    im_file = request.files["image"]
    im_bytes = im_file.read()
    im = Image.open(io.BytesIO(im_bytes))

    feature_list = []
    for i in range(len(id_feature_list)):
        feature_list.append(id_feature_list[i][1])

    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute',
                                 metric='euclidean').fit(feature_list)

    ss = get_vector(im.convert('RGB'))

    distances, indices = neighbors.kneighbors([ss])

    return json.dumps(get_ids_by_vectors(indices))


@app.route('/add', methods=['POST', 'GET'])
def add():
    with open('/home/fareshm/mysite/id_feature_list', 'rb') as f:
        id_feature_list = pickle.load(f)
    files = request.files.getlist("image")
    idd = request.form["idjj"]

    for im_file in files:
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        id_feature_list.append((int(idd), get_vector(im.convert('RGB'))))

    with open('/home/fareshm/mysite/id_feature_list', 'wb') as g:
        pickle.dump(id_feature_list, g)

    return json.dumps('success')


@app.route('/delete', methods=['POST', 'GET'])
def delete():
    with open('/home/fareshm/mysite/id_feature_list', 'rb') as g:
        id_feature_list = pickle.load(g)
    idd = request.form["idjj"]
    idd=int(idd)
    i = 0
    b = 0
    while True:
        if i == len(id_feature_list):
            break
        if id_feature_list[i][0] == idd:
            id_feature_list.pop(i)
            i = i - 1
            b=1
        i = i + 1

    with open('/home/fareshm/mysite/id_feature_list', 'wb') as f:
        pickle.dump(id_feature_list, f)
    if b==1:
        return json.dumps('success')
    else:
        return json.dumps('no delete')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
