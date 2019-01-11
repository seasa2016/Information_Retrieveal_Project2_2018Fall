# Relation Classification via LSTM
### Glove
please download the pretrain embedding {glove.6B.100d.txt} and put it under ./data/embedding

### Lstm model
##### - output at the last timestep
- python train.py --mode pretrain --model birnn
##### - With Nodes Information
- python train.py --mode pretrain --model birnn_co
##### - test
- python test.py --mode pretrain --model birnn_co --load_from {path to the .pt file}
### Evaluation
|Methods|Precision|Recall|Macro-F1|Micro-F1
|---|---|---|---|---
|output at end|95.54%|93.21%|94.30%|96.56%
|output at corresponding timestep|95.75%|95.27%|95.51%|97.65%


