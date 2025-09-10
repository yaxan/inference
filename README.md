```
python -m venv ./venv
source venv/bin/activate

uv pip install vllm --torch-backend=auto
pip install -r requirements.txt
```

```
hf auth login
```

```

cd ./vllm_srv
export VLLM_DOWNLOAD_DIR=~/hf-models   # optional cache location
export API_KEY=token-abc123            # any non-empty string works for local auth

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --download-dir "$VLLM_DOWNLOAD_DIR" \
  --api-key "$API_KEY"
```

```
export OPENAI_API_KEY=token-abc123
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Say hi in one short sentence."}],"max_tokens":32}'
```

```
cd ../harness
./client.py --concurrency 1  --out ../reports/baseline_c1.csv
./client.py --concurrency 4  --out ../reports/baseline_c4.csv
./client.py --concurrency 16 --out ../reports/baseline_c16.csv
```

```
python harness/summarize.py reports/baseline_c*.csv
```

```
./client.py --concurrency 4 --stream --out ../reports/baseline_stream_c4.csv
```