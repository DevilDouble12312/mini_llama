#!/usr/bin/env bash
# mini-llama serve 一键诊断脚本
# 用法（在 WSL 里，项目根目录下）：
#   bash scripts/serve_check.sh
# 作用：重新编译 -> 启动服务 -> 测 /health -> 测 /v1/generate
#       -> 把响应保存为 response.json 并校验 JSON 合法性。
set -u

cd "$(dirname "$0")/.."            # 切到项目根目录
ROOT="$(pwd)"
BUILD="$ROOT/build"
PORT="${PORT:-8090}"
MODEL="${MODEL:-models/tiny}"
OUT="$ROOT/response.json"

echo "==> 1/5 编译 mini-llama"
cmake --build "$BUILD" --target mini-llama -j"$(nproc)" || { echo "!! 编译失败，请把报错发给我"; exit 1; }

echo "==> 2/5 启动服务 (port $PORT, model $MODEL)"
"$BUILD/mini-llama" serve "$MODEL" --port "$PORT" >"$BUILD/serve.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; echo "==> 服务已停止"' EXIT

for _ in $(seq 1 30); do
  if grep -q "Server listening" "$BUILD/serve.log" 2>/dev/null; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "!! 服务进程启动后立即退出，日志如下："
    cat "$BUILD/serve.log"
    exit 1
  fi
  sleep 1
done
sleep 1

echo "==> 3/5 测 /health（不跑推理，应该秒回）"
if ! curl -s -m 5 -w "\nHTTP %{http_code} in %{time_total}s\n" "http://localhost:$PORT/health"; then
  echo "!! /health 无响应：服务可能没起来，或端口不通。serve.log 如下："
  cat "$BUILD/serve.log"
  exit 1
fi

echo "==> 4/5 测 /v1/generate，响应保存到 $OUT"
curl -s -m 60 -X POST "http://localhost:$PORT/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello","max_tokens":8}' \
  -w "\nHTTP %{http_code} in %{time_total}s\n" -o "$OUT"

echo "==> 5/5 校验 JSON"
if python3 -c "
import json
d = json.load(open('$OUT'))
print('response.json 是合法 JSON')
print('generated_text =', repr(d.get('generated_text')))
" 2>/dev/null; then
  :
elif command -v jq >/dev/null 2>&1 && jq . "$OUT" 2>/dev/null; then
  :
else
  echo "!! response.json 不是合法 JSON，文件内容："
  cat "$OUT"
fi

echo "==> 服务端日志（关键：应看到 [serve] POST /v1/generate -> 200 (xx ms)）"
tail -5 "$BUILD/serve.log"

echo ""
echo "==================================================================="
echo "如果上面全部成功，但你在 Windows 的 curl 还是没反应，原因基本是"
echo "Windows 访问不到 WSL2 的 localhost 转发，改用 WSL 的 IP 访问："
echo "  http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT"
echo "（服务还在后台跑着，测完会自动停止；response.json 在项目根目录）"
echo "==================================================================="
