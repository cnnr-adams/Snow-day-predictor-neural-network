import sys
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
from keras.models import load_model
from keras import backend as K
sys.stderr = stderr
import os
import sys
from flask import request
from flask import Flask
import numpy as np
app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
dir_name = os.path.dirname(os.path.realpath(__file__))


def predict(maxTemp, meanTemp, minTemp, rainFall, snowFall):
    K.clear_session()
    model = load_model(os.path.join(dir_name, "models/latest.h5"))
    return str(model.predict(np.array([[maxTemp, meanTemp, minTemp, rainFall, snowFall]]))[0][0].item())


@app.route("/")
def call():
    print('test', request.args, request.query_string)
    return predict(request.args.get("maxTemp"),
                   request.args.get("meanTemp"), request.args.get("minTemp"),
                   request.args.get("rainFall"), request.args.get("snowFall"))


if __name__ == '__main__':
    app.run(use_reloader=False, host='0.0.0.0')
