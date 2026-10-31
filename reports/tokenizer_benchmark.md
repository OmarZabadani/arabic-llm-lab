# Held-Out Tokenizer Benchmark

- Total sentences: 120
- Training sentences: 96
- Testing sentences: 24
- Training split: 80%
- Testing split: 20%
- Random seed: 42
- Custom BPE vocabulary size: 512

## Evaluation Results

| Tokenizer | Language | Words | Tokens | Tokens per word |
|---|---|---:|---:|---:|
| GPT-2 | Arabic | 124 | 833 | 6.718 |
| GPT-2 | English | 132 | 161 | 1.220 |
| Qwen2.5-0.5B | Arabic | 124 | 313 | 2.524 |
| Qwen2.5-0.5B | English | 132 | 159 | 1.205 |
| Custom BPE | Arabic | 124 | 517 | 4.169 |
| Custom BPE | English | 132 | 529 | 4.008 |
