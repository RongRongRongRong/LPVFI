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
1. Train LPVFI from scratch or use the pretrained model in './weights/LPVFI-IFRNet.pth'.
2. To test frame interpolation accuracy on Vimeo90K, UCF101, SNU-FILM and HD datasets, you can run:
```python
python test_MACs.py --dataset Vimeo90k --root 'data_path' --thres 15 --ckpt 'model_path'
python test_MACs.py --dataset UCF --root 'data_path' --thres 15 --ckpt 'model_path'
python benchmarks/SNU_FILM.py --thres 15 --root 'data_path' --ckpt 'model_path'
python benchmarks/HD.py --thres 15 --root 'data_path' --ckpt 'model_path'
```  




## To Do
Upload code about latency test on NVIDIA's CUTLASS Operator
