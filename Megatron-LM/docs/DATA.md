# Data preparation contracts

Data processing is separate from training. Every command receives source,
destination, tokenizer, worker count, shard range and manifest paths
explicitly; no cluster filesystem is compiled into Python or shell code.

## Common invariants

All stages use the OLMo 3 tokenizer contract:

```text
EOS = 100257
PAD = 100277
true vocabulary = 100278
padded vocabulary = 100352
```

For every prepared dataset, preserve a manifest containing source repository
and revision, tokenizer checksum, token dtype, number of documents/instances,
number of tokens and sharding plan. Stage 2 and Stage 3 runtime manifests also
require a SHA256 for every token array and paired metadata file; build their
inventories with `--checksums`.

Stage 1 training data is not reused as evaluation data. Evaluation manifests
are separate explicit inputs.

## Stage 1: Dolma 3 mix

The `dolma3_6t` profile expects Megatron indexed data and a generated
`data_args_path`.

Preparation steps:

1. download a pinned source revision into an externally supplied source directory;
2. build a deterministic tokenization shard plan;
3. tokenize shards with the supplied tokenizer;
4. produce indexed `.bin`/`.idx` artifacts and weighted data arguments;
5. validate token counts and dtype; the finalizer seals each `.idx` SHA256 and
   the corresponding `.bin` source contract and byte size;
6. pass `--data-root`, `--data-args-path` and `--tokenizer` to the launcher.

Stage 1 uses `split=100,0,0`: every selected training token stays in training.
The indexed `.bin`/`.idx` files are reusable directly. Set
`--data-cache-path` to one shared directory when compatible Stage 1 runs should
also reuse Megatron's derived sample-index files. The default is the isolated
`<output>/data-cache` directory.

Populate an explicit shared Stage 1 cache with one job before unrelated jobs
reuse it. Megatron synchronizes a cold-cache build within one distributed job,
not between independent jobs.

## Stage 2: Dolmino 100B

The `dolmino_100b` profile records source revision
`f23942ae8a8114af6e992efe8188ce8c531acd16` and known source token count
102,421,858,230. Training requests 100B tokens.

The FSL preparation contract:

- sequence length 8,192;
- deterministic data seed 1,337;
- drop an incomplete source tail;
- drop an incomplete global batch;
- write cache/index artifacts under an explicit work directory.

Runtime inputs are `--data-root`, `--data-manifest`, `--data-work-dir` and
`--tokenizer`.
The self-contained runtime manifest is mandatory; training will not infer a
source list or reuse a cache by directory convention. The rendered command
also seals the selected `configs/data/*.json` SHA256; runtime rejects a
manifest whose stage, backend, known token count, tokenizer IDs, packing
contract, profile name, or profile digest differs.

The FSL cache key contains the sealed runtime-manifest contract SHA256. Its
ordered source records contain content SHA256 values, so remapping a mount
point does not change cache identity and a new content contract cannot select
an older cache.

At launch, one control process per node performs a presence-and-size sweep
and validates the prepared cache completion/index contract before `torchrun`;
individual ranks do not rescan the shared filesystem.
Launch deliberately does not rehash hundreds of GiB of source data.
`olmo3_data.py verify --checksums` is the offline content audit after data
copy, repair, or suspected corruption.

## Stage 3: Longmino 50B

The `longmino_50b` profile records source revision
`8c0b3b265f95514c0f1b643c95da518e261a32a7` and known source token count
55,279,928,852. Training requests 50B tokens.

Packed preparation must emit:

- document lengths and compact `cu_seqlens`;
- deterministic source grouping/permutation;
- 65,536-token packed sequences;
- document-relative position IDs;
- terminal-target masks;
- enough boundary metadata for CP slicing and one-hop SWA halo;
- a content-sealed source manifest plus immutable completion metadata for every
  generated packed-boundary artifact.

Cross-document causal attention is forbidden even when documents share a
packed sequence. Stage 3 uses the same exact data-profile/manifest binding as
Stage 2, so a structurally valid Stage 2 or stale Longmino manifest cannot be
substituted silently.

## Stage 4: Think and Instruct SFT

Think and Instruct are prepared from separate immutable snapshots:

| Profile | Repository | Revision | Parquet files |
|---|---|---|---:|
| `dolci_think` | `allenai/Dolci-Think-SFT-7B` | `72ec0fe32428bada1ec686a9168a0681eecd2094` | 156 |
| `dolci_instruct` | `allenai/Dolci-Instruct-SFT` | `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221` | 15 |

Conversion uses the pristine official Open-Instruct checkout at commit
`78d1e5aa3cf80a73ce56fd0775e7ad959faf2660`. The plan expands the immutable
Parquet inventory into explicit `--dataset_mixer_list PATH 1.0` pairs. No
wildcard-support patch or other local edit is applied to Open-Instruct.

Each prepared directory provides:

- packed token IDs;
- labels with non-assistant tokens masked;
- document boundaries;
- a deterministic instance count;
- a content fingerprint.

The SFT fingerprint covers token arrays, assistant label masks, document
metadata, dataset statistics, and every file in the frozen tokenizer snapshot.
Preparation computes it once; distributed workers use the required expected
fingerprint and file-size checks rather than re-hashing the corpus per node.
After a data copy, repair, or suspected in-place mutation, rerun the offline
content audit and preparation before training.

The instance count and fingerprint are mandatory:

```text
--expected-instances <integer>
--expected-fingerprint <digest>
```

An executed `run-sft --report ...` report reads the packed cache's exact
`complete.json` and returns `instances`, `fingerprint`,
`packed_complete_manifest`, and `packed_complete_manifest_sha256`. The first
two values are the training assertions. Converted-file counts are not used as
instance-count substitutes.

The `sft-plan --work-root` value and training `--data-work-dir` value are the
same root. Both preparation and launch resolve its packed-cache parent as
`<work-root>/packed`; the per-node launch preflight verifies the prepared
instance count and fingerprint before `torchrun`.

The resolver uses the instance count and requested epochs to derive training
and warmup steps. A fingerprint mismatch must stop before training.

## Parallel preparation

Tokenization jobs can be partitioned by shard/part, but each part must have a
unique output and completion marker. Merge/index only after validating all
expected parts. A retry must be idempotent and must not silently append to an
existing artifact.

The work directory is either empty or bound to a compatible immutable
manifest. Cache identity is never inferred from a path belonging to another
experiment.
