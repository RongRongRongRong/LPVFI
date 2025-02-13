# LPVFI

## Python Environment   
python==3.7.16  
torch==1.9.1+cu111   
cupy-cuda111  

## Training
### Dataset Preparation
Download [Vimeo90k triplet dataset](http://toflow.csail.mit.edu/index.html#triplet), put the file into "./data".

### Flow Generation
Generate optical flow pseudo label to distill the model with `python generate_flow.py`

### Train LPVFI-IFRNet
Run train_vimeo90k.py.

## Test
1. Train LPVFI from scratch or download the pretrained model in this [link]().
2. To test frame interpolation accuracy on Vimeo90K, UCF101, SNU-FILM and HD datasets, you can run:
```python
python test_MACs.py --dataset Vimeo90k --thres 15  
python test_MACs.py --dataset UCF --thres 15  
python benchmarks/SNU_FILM.py 
python benchmarks/HD.py 
```  

