import base64
import requests

def encode(img_path):
    with open(img_path, 'rb') as reader:  # rb代表二进制加载
        img_content = reader.read()  # 加载图像的所有二进制数据
        img_base64_content = base64.b64encode(img_content)
    reader.close()
    return img_base64_content


def t0():
    r = requests.get(
        url="http://127.0.0.1:9001/predict",
        params={
            'path': '/mnt/code/shenlan/project1/pro01/datas/MNIST/MNIST/test/1/48040.png',
            'topk': 3
        }
    )
    if r.status_code == 200:
        print("请求服务器并成功返回")
        print(r.json())
        print(r.text)
    else:
        print(f"请求服务器网络异常：{r.status_code}")



if __name__ == '__main__':
    t0()