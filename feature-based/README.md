# Feature-based Approach
## Features
- Nodes information (e.g. value, agent, condition)
- Number of words/characters/capital letters...
## Classification
- Multiclass (3 kinds of labels for edge)
- Catboost classifier
## Execution
- Set file directory: train.json, test.json
- Install Catboost package
- python3 classification.py
## Evaluation
|Methods|Precision|Recall|Macro-F1|Micro-F1
|---|---|---|---|---
|Rule-based|0.9319|0.9516|0.9411|0.9747
|Feature-based|0.9667|0.9467|0.9548|0.9819
