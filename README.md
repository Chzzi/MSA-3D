# Multi-Scale Attention 3D Convolutional Network for Multimodal Gesture Recognition

The pytorch implement of "[Multi-Scale Attention 3D Convolutional Network for 
Multimodal Gesture Recognition](https://www.mdpi.com/1424-8220/22/6/2405)"

## The environment

- Python 3.7 :: Anaconda
- Pytorch 1.11.0
- Numpy
- OpenCV
- PIL 

This is the enviroment we used, you can install Python 3.* and Pytorch 1.* instead. 

## To prepare the data
This experiment based on [Chalearn LAP IsoGD Database](http://www.cbsr.ia.ac.cn/users/jwan/database/isogd.html) and a dataset derive from IsoGD by using hand detector namely IsoGD_hand
```
│IsoGD/                             │IsoGD_hand/
├──train/                           ├──train/
│  ├── 001                          │  ├── 001
│  │   ├── K_00001.avi              │  │   ├── K_00001.avi
│  │   ├── K_00002.avi              │  │   ├── K_00002.avi
│  │   ├── ......                   │  │   ├── ......
│  ├── 002                          │  ├── 002
│  │   ├── K_00201.avi              │  │   ├── K_00201.avi
│  │   ├── K_00202.avi              │  │   ├── K_00202.avi
│  │   ├── ......                   │  │   ├── ......
│  ├── ......                       │  ├── ......
├──valid/                           ├──valid/
├──test/                            ├──test/
├──train.txt                        ├──train.txt
├──valid.txt                        ├──valid.txt
├──test.txt                         ├──test.txt
```
## Train global branch

To train MSA-3D gobal branch
```bash
python train_MSA3D_global.py --gpu {gpu index} --data_root {train_folder_path of IsoGD} --hand_data_root {train_folder_path of IsoGD_hand}\
 --ground_truth {train_ground_truth of IsoGD} --test_data_root {test_folder_path of IsoGD} --test_hand_root {test_folder_path of IsoGD_hand}\
 --test_ground_truth {test_ground_truth of IsoGD} --save_dir {path of save model}
```
To save the feature of global branch (path of feature file should be end of *.pkl)
```bash
python feature_global_output.py --test_data_root {test_folder_path of IsoGD} --test_hand_root {test_folder_path of IsoGD_hand}\
 --test_ground_truth {test_ground_truth of IsoGD} --model_path {path of saved model} --feature_save_path {feature file path will be saved}
```

## Train depth branch
To train MSA-3D depth branch
```bash
python train_MSA3D_depth.py --gpu {gpu index} --data_root {train_folder_path of IsoGD} --ground_truth {train_ground_truth of IsoGD} --test_data_root {test_folder_path of IsoGD} --test_ground_truth {test_ground_truth of IsoGD} --save_dir {path of save model}
```
To save the feature of depth branch (path of feature file should be end of *.pkl)
```bash
python feature_depth_output.py --test_data_root {test_folder_path of IsoGD} --test_ground_truth {test_ground_truth of IsoGD} --model_path {path of saved model} --feature_save_path {feature file path will be saved}
```

## Train hand branch
To train MSA-3D hand branch
```bash
python train_MSA3D_hand.py --gpu {gpu index} --data_root {train_folder_path of IsoGD_hand} --ground_truth {train_ground_truth of IsoGD_hand} --test_data_root {test_folder_path of IsoGD_hand} --test_ground_truth {test_ground_truth of IsoGD_hand} --save_dir {path of save model}
```
To save the feature of hand branch (path of feature file should be end of *.pkl)
```bash
python feature_hand_output.py --test_data_root {test_folder_path of IsoGD_hand} --test_ground_truth {test_ground_truth of IsoGD_hand} --model_path {path of saved model} --feature_save_path {feature file path will be saved}
```

## Score fusion
Get the final result by fusing global branch features, depth branch features, and hand branch features.
```bash
python score_fusion.py --global_feature_path {feature path of global branch} --depth_feature_path {feature path of depth branch} --hand_feature_path {feature path of hand branch} --groud_truth_path {test_ground_truth of IsoGD}
```

# Citation
```
@inproceedings{chen2022MSA3D,
      title={Multi-Scale Attention 3D Convolutional Network for Multimodal Gesture Recognition}, 
      author={Huizhou Chen and Yunnan Li and Huijuan Fang and Wentian Xin and Zixiang Lu and Qiguang Miao},
      journal={Sensors},
      year={2022},
}
```