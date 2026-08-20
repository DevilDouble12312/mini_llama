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

-------使用Qwen2 模型测试能否正常对话------

## 模型下载

由于模型文件较大（约 530 MB），未直接存放在本仓库中。请使用以下方式之一下载，并将文件放入 `models/` 目录。

### 推荐方式：使用 huggingface-cli

1. 安装 `huggingface-hub`：
   ```bash
   pip install huggingface-hub
   ```
2. 在项目根目录执行下载命令：
   ```bash
   huggingface-cli download msimou/Qwen2-0.5B-Instruct-GGUF Qwen2-0.5B-Instruct-Q8_0.gguf --local-dir ./models
   ```

### 备选方式：从 Hugging Face 官网下载

访问 [msimou/Qwen2-0.5B-Instruct-GGUF](https://huggingface.co/msimou/Qwen2-0.5B-Instruct-GGUF) 页面，手动下载 `Qwen2-0.5B-Instruct-Q8_0.gguf` 文件，并放入 `models/` 文件夹。

下载完成后:

重新cmake ..

make

./mini-llama serve models/chat/qwen2-0_5b-instruct-q8_0.gguf --port 8080


再次进入网站,发现可以交谈
