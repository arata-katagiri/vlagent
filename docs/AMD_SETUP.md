# AMD integration runbook

Goal: the skill-writing model runs on an AMD MI300X via vLLM, the repo talks to it through the
existing `OPENAI_BASE_URL` switch, and best-of-N (N=4) candidate skills are generated in one batched
call. Budget: 40 minutes if sign-up is instant. Hard stop: if no endpoint answers by 14:45, abandon
and stay on OpenRouter. Nothing in the main build depends on this.

## 0. Tell Timothy (2 min)
In person, now. He submits the team name to AMD. Without this nothing else here matters.

## 1. Credits and a GPU droplet (10-15 min)
1. With the AI Developer Program account, the $100 credit is **applied automatically** to the
   AMD Developer Cloud account (devcloud.amd.com, DigitalOcean-backed). No claim form. Check
   Billing in the dashboard: the credit balance should show there. It expires 30 days after
   it lands.
2. Billing → add a payment method. Required before any GPU droplet can be created; credits are
   consumed first and the card only covers overage.
3. GPU Droplets → Create GPU Droplet:
   - Region: **ATL1 (Atlanta)** has MI300X stock.
   - Plan: **MI300X (1 GPU)**, 192 GB, $1.99/h.
   - Image: **vLLM Quick Start Package** (OpenAI-compatible API preconfigured).
   - SSH key: upload your public key.
   Provisioning takes 2-4 minutes; wait for status Active and note the IP.

## 2. Serve the model (10-15 min, mostly weight download)
Model: `Qwen/Qwen2.5-Coder-32B-Instruct`. Not gated, no HF token needed, ~65 GB bf16, strong at
Python, tool calling works with the `hermes` parser. Fits one MI300X with room to spare.

```bash
ssh root@<IP>
# The quick-start image ships a ROCm container named `rocm` with vLLM inside:
docker exec -it rocm /bin/bash
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
    --dtype bfloat16 --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --api-key vlagent-demo --port 8000 --host 0.0.0.0
# (run it under nohup or tmux so it survives the SSH session)

# Only if the image has no such container, run vLLM yourself:
docker run -d --name vllm \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g \
  -p 8000:8000 rocm/vllm:latest \
  vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
    --dtype bfloat16 --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --api-key vlagent-demo --port 8000
docker logs -f vllm      # wait for "Application startup complete"
```
If the quick-start image already runs vLLM as a service, skip docker and just start it with the
same flags. `rocm-smi` shows the card; screenshot it for the submission.

## 3. Reach it safely: SSH tunnel, do not open the port (1 min)
On the demo Mac, keep this running in a spare terminal:
```bash
ssh -N -L 8000:localhost:8000 root@<IP>
```
Test:
```bash
curl -s http://localhost:8000/v1/models -H "Authorization: Bearer vlagent-demo" | head -c 300
```

## 4. Point the repo at it (1 min)
`.env`:
```
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=vlagent-demo
OPENAI_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
```
Smoke test (must print a plan, not a traceback):
```bash
printf 'Put the cup on the tray.\nquit\n' | python -m robot_agent.app.main --no-viewer --no-log
```
If `tool_choice="required"` is rejected by this vLLM build, change it to `"auto"` in
`robot_agent/agent/llm.py` for the AMD run; the prompt already forces a tool call.

## 5. Best-of-N in the client (the only code change, ~15 lines)
In `LLMClient`, the `define_skill` call passes `n=4` and returns every choice's tool call instead of
`choices[0]`. vLLM batches the 4 samples in one forward pass. Use `temperature=0.8` for diversity.
The skills layer rehearses each candidate, scores by (effect achieved, side effects, off-table),
and keeps the best. Against OpenRouter the same code still works (`n` is passed through for
OpenAI models), so this is not AMD-only; AMD is what makes it cheap enough to always do.

```python
resp = self.client.chat.completions.create(
    model=self.model, messages=messages, tools=tools,
    tool_choice="required", n=4, temperature=0.8, max_tokens=1200)
candidates = [json.loads(c.message.tool_calls[0].function.arguments)
              for c in resp.choices if c.message.tool_calls]
```

## 6. Make it verifiable for AMD's remote check (5 min)
- Session log: add the endpoint host to `RunLog.write("session", ...)` in `app/main.py`
  (`urlparse(os.environ["OPENAI_BASE_URL"]).hostname`). Commit a run log from the AMD session in
  `runs/` (they are gitignored; force-add one).
- README section "AMD Developer Cloud": droplet type, model, vLLM flags, the `rocm-smi` screenshot,
  and the sentence "skill candidates are generated 4-wide on an MI300X and rehearsed in MuJoCo".
- Written submission: state AMD infrastructure explicitly, as the handbook requires.

## 7. Shut it down after the video
`docker stop vllm` and destroy the droplet, or the credits keep draining at $1.99/h.

## Fallbacks
- Sign-up needs approval → stop, stay on OpenRouter, still ship best-of-N with `n=4`.
- 32B download too slow → `Qwen/Qwen2.5-Coder-7B-Instruct` (15 GB) is fine for the demo.
- Tool parsing flaky → drop `--enable-auto-tool-choice`, set `tool_choice="none"`, and have the
  model return JSON in content; the planner already tolerates a content-only answer.
