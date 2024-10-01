import os

# 指定文件夹路径
folder_path = r'E:\opencv_sources-4.9.0\build\install\x64\vc16\lib'

# 获取所有.dll文件名
dlib_files = [f for f in os.listdir(folder_path) if f.endswith('.lib')]

# 打印文件名
for dlib_files in dlib_files:
    print(dlib_files)
lib_files = [f for f in os.listdir(folder_path) if f.endswith('.lib')]
# 打印文件名
# for i in range(len(lib_files)):
#     if lib_files[i].endswith('d.lib'):
#         continue
#     print(lib_files[i])
