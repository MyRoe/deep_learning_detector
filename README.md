# How to run

My environment is python==3.8.5  torchvision==0.8.1  pytorch==1.7.0  

```python
#train
python train.py  --mode="train"
#test
python train.py  --mode="test"
```

# Dataset

You can download  BOSSbase dataset [here](http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip).

Training and validation must contains there own cover and stego images directory, stego images must have the same name than there corresponding cover.

