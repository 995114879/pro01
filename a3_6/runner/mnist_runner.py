import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))


def tt_dl_model_v0():
    def training_model():
        from a3_6.dl_v0 import training
        import time
        # 开始计时
        start_time = time.time()
        training.run(
            img_path_dir="../../datas/cat_vs_dog",
            total_epoch=100, batch_size=8,
            model_output_path="./output/cat_vs_dog/dl_v0/model.pkl",
            last_model_path="./output/cat_vs_dog/dl_v0/last.pkl",
            best_model_path="./output/cat_vs_dog/dl_v0/best.pkl",
            scripted_model_output_path="./output/cat_vs_dog/pt/model_scripted.pt",
            log_file_path="train_log.txt",
            stop_threshold=0.01,  # 损失变化阈值
            patience=5,  # 提前停止等待的epoch数
            use_gpu=True,
            tensorboard_log_dir="./tensorboard_logs",  # TensorBoard日志目录
            load_best_model = True  # 这里加载最佳模型
        )
        # 结束计时
        end_time = time.time()
        # 计算并打印耗时（毫秒）
        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"Elapsed time: {elapsed_time_ms:.3f} ms")  # CPU Elapsed time: 475102.205 ms  485700.095 ms
        # 将结果保存到DataFrame
        df = pd.DataFrame([elapsed_time_ms], columns=["Elapsed Time (ms)"])
        # 将DataFrame保存到Excel文件
        df.to_excel('model_performance.xlsx', index=False)

    def predict_model():
        from a3_6.dl_v0 import predictor
        p = predictor.Predictor(
            algo_path=r"output/MNIST/dl_v0/model.pkl"
        )
        r = p.predict(
            img_path=r"../../datas/MNIST/MNIST/test/0/48006.png",
            k=3
        )
        print(r)

    def client_model_with_shell():
        from a3_6.dl_v0 import shell_client
        shell_client.run(
            algo_path=r"output/MNIST/dl_v0/model.pkl"
        )

    def start_flask_api():
        from a3_6.dl_v0 import flask_app
        flask_app.app.run()

    training_model()
    # predict_model()
    # client_model_with_shell()
    # start_flask_api()


if __name__ == '__main__':
    tt_dl_model_v0()
