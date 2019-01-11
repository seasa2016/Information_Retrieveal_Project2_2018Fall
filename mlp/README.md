# Relation Extraction with MLP (Multilayer Perceptron)
## Word Embedding
- Download: 'GoogleNews-vectors-negative300.bin'
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
- Set the directory for the pre-trained word embedding
## Execution
- Set file directory: train.json, test.json
- Without node info:
> python3 mlp_wo_node.py
- With node info: 
> python3 mlp_w_node.py
## Evaluation
|Methods|Precision|Recall|Macro-F1|Micro-F1
|---|---|---|---|---
|MLP (without node info)|0.9267|0.9300|0.9300|0.9561
|MLP (with node info)|0.9733|0.9500|0.9606|0.9843
