# TokenCluster

This repository presents the implementation of the ACL 2023 findings paper:

Aspect-aware Unsupervised Extractive Opinion Summarization





## Data

Download the SPACE corpus from this [link](https://github.com/stangelid/qt).
Amazon dataset is publicly available [here](https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer/tree/master/gold_summs).



## Inference Using SemAE

On Amazon, run the following:
```
sh scripts/infer_amazon.sh
```

On SPACE, for general summarization, run the following:
```
sh scripts/infer_space.sh
```


On SPACE, for aspect-specific summarization, run the following:
```
sh scripts/asp_infer_space.sh
```

## Inference Using TokenCluster

On Amazon, run the following:
```
sh scripts/infer_amazon_tk.sh
```

On SPACE, for general summarization, run the following:
```
sh scripts/infer_space_tk.sh
```


On SPACE, for aspect-specific summarization, run the following:
```
sh scripts/asp_infer_space_tk.sh
```
