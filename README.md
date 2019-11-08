# ASC_failure
code for the paper: [A Failure of Aspect Sentiment Classifiers and an Adaptive Re-weighting Solution](http://arxiv.org/abs/1911.01460).

### Problem to Solve
We found Aspect Sentiment Classifier does poorly on ASPECT-LEVEL sentiment classification, because most training examples have the same sentence-level and aspect-level polarity. Sentences with more than 1 opinions of different polarities (we call them contrastive sentences) are rare but truly indicating aspect-level polarity.

### Dataset
We leverage the dataset from SemEval 2014 but augment the testing set of laptop with more contrastive sentences to test aspect-level sentiment. The dataset is self-contained in this repository.

### Environment
This code base is tested on GTX 1080 Ti, with Ubuntu 16.04, Python3.6, PyTorch 1.0.1 and pytorch transformer 0.4 ```pip install pytorch-pretrained-bert==0.4.0```.

### Training

Download weights of BERT-DK:  
[laptop](https://drive.google.com/file/d/1TRjvi9g3ex7FrS2ospUQvF58b11z0sw7/view?usp=sharing) to ```pt_model/laptop_pt_review```  
[restaurant](https://drive.google.com/file/d/1nS8FsHB2d-s-ue5sDaWMnc5s2U1FlcMT/view?usp=sharing) to ```pt_model/rest_pt_review```  

```
bash run.sh train
```

### Evaluation

```
bash run.sh test
```

### Citation
```
@article{xu2019afailure,
  title={A Failure of Aspect Sentiment Classifiers and an Adaptive Re-weighting Solution},
  author={Xu, Hu and Liu, Bing and Shu, Lei and Yu, Philip S},
  journal={arXiv preprint arXiv:1911.01460},
  year={2019}
}
```
