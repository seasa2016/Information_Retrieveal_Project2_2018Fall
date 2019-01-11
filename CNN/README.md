# Relation Classification via CNN
### Word2vec model
### Features
##### -Lexical Level Features
##### -Sentence Level Features
- Word features
- Position features
### CNN model
##### -Without Nodes Information
- python3 cnn.py
- python3 cnn_test.py
##### -With Nodes Information
- python3 cnn_nodes.py
- python3 cnn_nodes_test.py
### Evaluation
|Methods|Precision|Recall|Macro-F1|Micro-F1
|---|---|---|---|---
|CNN (without Nodes Info)|83.65%|58.71%|61.60%|76.34%
|CNN (with    Nodes Info)|96.31%|92.94%|94.45%|97.83%
### Reference
- Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, and Jun Zhao. 2014. Relation classification via convolutional deep neural network. In Proceedings of COLING, pages 2335–2344.
