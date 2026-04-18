---
dataset_info:
  features:
  - name: tokens
    sequence: string
  - name: ner_tags
    sequence: string
  splits:
  - name: train
    num_bytes: 6271422
    num_examples: 15023
  - name: validation
    num_bytes: 667453
    num_examples: 1669
  - name: test
    num_bytes: 784028
    num_examples: 1854
  download_size: 1595282
  dataset_size: 7722903
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

['CELL_LINE', 'CELL_TYPE', 'DNA', 'PROTEIN', 'RNA']