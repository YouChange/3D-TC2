This is the official implementation of 3D-TC2(Temporal Consistency Checks to Detect LiDAR Spoofing Attacks on Autonomous Vehicle Perception)
Paper Link: `https://dl.acm.org/doi/pdf/10.1145/3469261.3469406`

## Requirements
- CUDA >= 9.0
- Python 3
- PyTorch >= 1.1
- pyquaternion, Matplotlib, PIL, numpy, cv2, tqdm, scipy, scikit-image, scikit-learn, ipython and other relevant packages

## Usage
1. Add path to the root folder. For example:
```
export PYTHONPATH=/home/cy19/homedir/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/cy19/homedir/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH
```

2. Dataset
Download Nuscenes dataset from `https://www.nuscenes.org/`. For example, to download the nuScenes mini split:
`wget https://www.nuscenes.org/data/v1.0-mini.tgz `

Dataset folder: ./data/nuscenes/

3. Attack the dataset for object detection
In our paper, we assume historical scenes are not poisoned and performed single-frame injection attack. 

You can also customize your own poisoned LiDAR dataset via other attack methods(e.g., consecutive attacks) to perform stress tests on 3D-TC2.

4. Object detection
Please feed your poisoned dataset to any kind of 3D object detectors and get predictions.

Our detection results after running OpenPCDet(https://github.com/open-mmlab/OpenPCDet.git) can be downloaded here: link

5. Object-Motion prediction
Use pretrained MotionNet(model.pth) to detect anomalies based on historical scenes:
```
python plots.py --data ./data/nuscenes/mini/ --version v1.0-mini --modelpath model.pth --net MotionNet --savepath log
```


## Reference
@inproceedings{you2021temporal,
  title={Temporal Consistency Checks to Detect LiDAR Spoofing Attacks on Autonomous Vehicle Perception},
  author={You, Chengzeng and Hau, Zhongyuan and Demetriou, Soteris},
  booktitle={Proceedings of the 1st Workshop on Security and Privacy for Mobile AI},
  pages={13--18},
  year={2021}
}