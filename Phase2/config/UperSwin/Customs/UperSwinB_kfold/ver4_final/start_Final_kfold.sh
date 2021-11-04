# python3 /opt/ml/segmentation/mmsegmentation/tools/train.py /opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/Augs/uper_swinS_augs1.py --seed 2021
cd /opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/Customs/UperSwinB_kfold/ver4_final
python /opt/ml/segmentation/mmsegmentation/tools/train.py /opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/Customs/UperSwinB_kfold/ver4_final/fold1.py --seed 2021
python /opt/ml/segmentation/mmsegmentation/tools/train.py /opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/Customs/UperSwinB_kfold/ver4_final/fold2.py --seed 2021
python /opt/ml/segmentation/mmsegmentation/tools/train.py /opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/Customs/UperSwinB_kfold/ver4_final/fold3.py --seed 2021
python /opt/ml/segmentation/mmsegmentation/tools/train.py /opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/Customs/UperSwinB_kfold/ver4_final/fold4.py --seed 2021
python /opt/ml/segmentation/mmsegmentation/tools/train.py /opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/Customs/UperSwinB_kfold/ver4_final/fold5.py --seed 2021