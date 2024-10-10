from flask import Flask, request, jsonify
from a3_5.dl_v0 import predictor

app = Flask(__name__)
p = predictor.Predictor(
    algo_path='./output/MNIST/dl_v0/model.pkl'
)


@app.route("/")
@app.route("/index")
def index():
    return "Flask App Model Predictor Demo"


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    允许给定url路径或者path路径
    url表示网络上可以获得的图像路径
    path表示服务器本地的路径
    :return:
    """
    try:
        _url, _path, _img, _k = None, None, None, 1
        if request.method == "GET":
            _args = request.args
            print(_args)
            _url = _args.get('url')
            _path = _args.get('path')
            _k = int(_args.get('topk', '1'))
        else:
            _args = request.form
            _img = _args.get('image')
            _k = int(_args.get('topk', '1'))
        r = p.predict(img_path=_path, img_url=_url, base64_img=_img, k=_k)
        return jsonify(r)
    except Exception as e:
        return jsonify({"code": 2, 'msg': f"服务器异常：{e}"})


def run():
    app.run(host='0.0.0.0', port=9001)
