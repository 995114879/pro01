import time

# 开始计时
start_time = time.time()

# 这里放置你想要计时的代码
# 例如:
for _ in range(1000000):
    pass

# 结束计时
end_time = time.time()

# 计算并打印耗时（毫秒）
elapsed_time_ms = (end_time - start_time) * 1000
print(f"Elapsed time: {elapsed_time_ms:.3f} ms")
