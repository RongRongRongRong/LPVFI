# LPVFI

## python environment   
python==3.7.16  
torch==1.9.1+cu111   
cupy-cuda118  

## training
### dataset prapre
Download [Vimeo90k triplet dataset](http://toflow.csail.mit.edu/index.html#triplet), put the file into "./data".

### train LPVFI-IFRNet
Run train_vimeo90k.py.

## test
Train LPVFI from scratch or download the pretrained model in this [link]()  

### test Vimeo90k dataset
run test_MACs.py with
'''bash
python test_MACs.py --dataset Vimeo90k

