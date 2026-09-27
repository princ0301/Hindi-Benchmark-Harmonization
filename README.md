# Hindi benchmark harmonization

## Running the remote Gemma model

`google/gemma-2-9b-it` is a gated Hugging Face repository. Before the first
Modal run, accept its access terms while signed in to the Hugging Face account
that owns your token, then create the Modal secret:

```powershell
modal secret create huggingface --from-literal HF_TOKEN=hf_your_token_here
uv run modal run src/pipeline/remote_gpu.py
```

The application injects this secret only into the remote GPU container; it is
not included in the source code or model volume.
