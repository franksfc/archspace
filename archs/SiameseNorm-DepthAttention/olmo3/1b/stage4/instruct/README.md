# Stage 4 Instruct SFT checkpoint

This subfolder contains a complete Hugging Face checkpoint with a context
length of 65536 tokens. Load it from the parent model repository using
`trust_remote_code=True`, `subfolder="olmo3/1b/stage4/instruct"`, and
`attn_implementation="sdpa"`.
