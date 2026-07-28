# Stage 2 mid-training checkpoint

This subfolder contains a complete Hugging Face checkpoint with a context
length of 8192 tokens. Load it from the parent model repository using
`trust_remote_code=True`, `subfolder="olmo3/1b/stage2"`, and
`attn_implementation="sdpa"`.
