# Held-Out Tokenizer Benchmark

- Arabic articles: 26,416
- English sentences: 60
- Arabic training articles: 21,133
- English training sentences: 48
- Arabic testing articles: 5,283
- English testing sentences: 12
- Training split: 80%
- Testing split: 20%
- Random seed: 42
- Custom BPE vocabulary size: 512

## Evaluation Results

| Tokenizer | Language | Words | Tokens | Tokens per word |
|---|---|---:|---:|---:|
| GPT-2 | Arabic | 3,099,193 | 17,892,050 | 5.773 |
| GPT-2 | English | 132 | 161 | 1.220 |
| Qwen2.5-0.5B | Arabic | 3,099,193 | 6,985,580 | 2.254 |
| Qwen2.5-0.5B | English | 132 | 159 | 1.205 |
| Custom BPE | Arabic | 3,099,193 | 10,029,045 | 3.236 |
| Custom BPE | English | 132 | 956 | 7.242 |
