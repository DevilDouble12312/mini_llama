//-----tiny测试----------
------在wsl环境下编译运行----

1, 进入文件目录下, 使用cmake外部编译

mkdir build 

cd build

cmake ..

make 

2, 使用接口监听

./mini-llama serve models/tiny --port 8080

找到输出的Web UI:    http://localhost:8080/

![Uploading image.png…]()

