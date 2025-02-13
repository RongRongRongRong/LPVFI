# LPVFI-AMT

##Python Environment
Create Conda Environment with `conda env create -f environment.yaml`

##Train
### Dataset Preparation
Download [Vimeo90k triplet dataset](http://toflow.csail.mit.edu/index.html#triplet)

###Flow Generation
Generate optical flow pseudo label to distill the model with `python generate_flow.py -r 'path/data/vimeo_triplet'`

### Train LPVFI-AMT
1.Change the 'dataset_dir' and 'flow_path' in LPVFI-AMT-S.yaml  
2.Run train.py.

##Test
1. Train LPVFI from scratch or download the pretrained model in this [link]().  
2. To test frame interpolation accuracy on Vimeo90K, UCF101, SNU-FILM and HD datasets, you can run:
```python
python test_MACs.py --dataset Vimeo90k --thres 15  
python test_MACs.py --dataset UCF --thres 15  
python benchmarks/SNU_FILM.py 
python benchmarks/HD.py 
```  