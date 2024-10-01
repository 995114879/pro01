import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))


def tt_dl_model_v0():
    def training_model():
        from a3_5.dl_v0 import training
        import time
        # 开始计时
        start_time = time.time()
        training.run(
            img_path_dir="../../datas/MNIST/MNIST/train",
            total_epoch=1, batch_size=8,
            model_output_path="./output/MNIST/dl_v0/model.pkl",
            use_gpu=True
        )
        # 结束计时
        end_time = time.time()
        # 计算并打印耗时（毫秒）
        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"Elapsed time: {elapsed_time_ms:.3f} ms")  # CPU Elapsed time: 1189675.224 ms
        # 将结果保存到DataFrame
        df = pd.DataFrame(elapsed_time_ms)
        # 将DataFrame保存到Excel文件
        df.to_excel('model_performance.xlsx', index=False)

    def predict_model():
        from a3_5.dl_v0 import predictor
        p = predictor.Predictor(
            algo_path=r"output/MNIST/dl_v0/model.pkl"
        )
        r = p.predict(
            img_path=r"../../datas/MNIST/MNIST/test/0/48006.png",
            k=3
        )
        print(r)

    # training_model()
    predict_model()


if __name__ == '__main__':
    tt_dl_model_v0()
