# ASC_failure
code for the paper about the failure of Aspect Sentiment Classifier (ASC).

### Problem to Solve
We found Aspect Sentiment Classifier does poorly on aspect-level, which is caused by rare contrastive sentences.

### Training

Download weights of BERT-DK:
[laptop](https://drive.google.com/file/d/1TRjvi9g3ex7FrS2ospUQvF58b11z0sw7/view?usp=sharing) to ```pt_model/laptop_pt_review```  
[restaurant](https://drive.google.com/file/d/1nS8FsHB2d-s-ue5sDaWMnc5s2U1FlcMT/view?usp=sharing) ) to ```pt_model/rest_pt_review```

```
bash run.sh train
```

### Evaluation

```
bash run.sh test
```
