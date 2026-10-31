#!/bin/bash
# Qwen3.5-35B-A3B API 测试脚本
PORT=${1:-8000}
BASE="http://localhost:$PORT/v1/chat/completions"
MODEL="qwen35-35b"

# ---- 测试 1: 简单问答 ----
echo "=== 测试 1: 简单问答 ==="
curl -s "$BASE" -H "Content-Type: application/json" -d '{
  "model": "'$MODEL'",
  "messages": [{"role": "user", "content": "你好，用一句话介绍你自己"}],
  "max_tokens": 800
}' | python3 -c "
import json, sys
d = json.load(sys.stdin)
if 'choices' in d:
    print(d['choices'][0]['message']['content'])
elif 'error' in d:
    print('ERROR:', d['error']['message'])
"

echo ""
echo "=== 测试 2: 代码生成 ==="
curl -s "$BASE" -H "Content-Type: application/json" -d '{
  "model": "'$MODEL'",
  "messages": [{"role": "user", "content": "写一个 Python 函数计算斐波那契数列"}],
  "max_tokens": 1500
}' | python3 -c "
import json, sys
d = json.load(sys.stdin)
if 'choices' in d:
    print(d['choices'][0]['message']['content'])
elif 'error' in d:
    print('ERROR:', d['error']['message'])
"

echo ""
echo "=== 测试 3: 翻译 ==="
curl -s "$BASE" -H "Content-Type: application/json" -d '{
  "model": "'$MODEL'",
  "messages": [{"role": "user", "content": "将以下中文翻译成英文：人工智能正在改变世界。"}],
  "max_tokens": 800
}' | python3 -c "
import json, sys
d = json.load(sys.stdin)
if 'choices' in d:
    print(d['choices'][0]['message']['content'])
elif 'error' in d:
    print('ERROR:', d['error']['message'])
"
