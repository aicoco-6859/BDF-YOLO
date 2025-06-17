# BDF-YOLO: A Robotic Grasping and Segmentation Network of Box-Shaped Objects Based on RGBD Feature Fusion
Box-shaped object sorting constitutes a critical task within industrial automation. Effectively fusing multi-modal features to enable rapid and accurate grasping of box-shaped objects in complex stacked scenarios remains a challenging problem. To address this issue, this work proposes the Bi-modal Dynamic Fusion YOLO (BDF-YOLO) instance segmentation network, which comprises a Dynamic Band-Channel Attention (DBCA) module, along with a Dynamic Cross Fusion Module (DCFM) incorporating the Dynamic Multi-scale Cascaded Attention Module (DMCAM). Based on the segmentation results from BDF-YOLO, grasp parameters are generated through template matching and pose estimation, which then drive robotic arms to complete the sorting task. On the synthetic dataset SYN-PBOX, BDF-YOLO achieves an mAP50 of 97% and an mAP50-95 of 93.2%, outperforming state-of-the-art methods by 1.2% and 3.1%, respectively. The sorting system trained on this dataset demonstrates a 97% grasping success rate, with an average center point error of 8.5 mm, an average plane normal vector error of 7.3°, and an average object orientation error of 6.2°, indicating that BDF-YOLO effectively addresses collaborative sorting requirements for box-shaped objects in complex stacked environments. Furthermore, our method achieves an mAP50-95 of 67.3% on the LLVIP infrared dataset, outperforming the second-best method by 2.0%.

**The source code will be submitted after the paper is accepted.**
___
# Frame diagram
![抓取流程图(英文版)](https://github.com/user-attachments/assets/cfd74cb5-6cd0-474d-beea-c01e0d6ece04)
___
# Grasping demo video:
## bilibili
___
# Datasets Download
## Baidu pan
链接: https://pan.baidu.com/s/1BiAE-sitRzVneWt7Tt6khg?pwd=d3i5 提取码: d3i5 
___
# Experimental index
## Results on SYN-PBOX
| Method          | Modality | mAP50  | mAP50-95 |
| :-------------: | :------: | :----: | :------: |
| (2021)CFT       | RGBD     | 0.953  |  0.896   |
| (2023)SuperYOLO | RGBD     | 0.954  |  0.886   |
| (2023)YOLOv8m   | RGBD     | 0.957  |  0.895   |
| (2024)YOLOv10m  | RGBD     | 0.955  |  0.898   |
| (2024)YOLO11m   | RGBD     | 0.951  |  0.889   |
| (2024)BGF-YOLO  | RGBD     | 0.847  |  0.770   |
| (2025)YOLOv12m  | RGBD     | 0.953  |  0.895   |
| (2025)AS-YOLO   | RGBD     | 0.856  |  0.786   |
| (2025)DEYOLO    | RGBD     | 0.951  |  0.890   |
| (2025)DS-YOLO   | RGBD     | 0.958  |  0.901   |
| **BDF-YOLO(ours)**        | **RGBD** | **0.970** | **0.932** |
## Grasping Performance
| Method            | Modality | Center point error(mm) | Normal vector error (°) | Angular error (°) | Grasping success rate (%) |
| :---------------: | :------: | :-------------------: | :---------------------: | :---------------: | :-----------------------: |
| (2023)YOLOv8m     | RGBD     | 9.2                  | 14.7                   | 11.6             | 90.5                     |
| (2024)BGF-YOLO    | RGBD     | 8.9                  | 15.2                   | 10.4             | 89.2                     |
| (2024)YOLOv10m    | RGBD     | 9.3                  | 15.3                   | 11.1             | 90.4                     |
| (2024)YOLO11m     | RGBD     | 8.7                  | 14.5                   | 9.7              | 89.6                     |
| (2025)YOLOv12m    | RGBD     | 9.1                  | 14.1                   | 9.8              | 91.0                     |
| (2025)AS-YOLO     | RGBD     | 8.8                  | 15.1                   | 10.1             | 89.5                     |
| (2025)DEYOLO      | RGBD     | 9.2                  | 14.8                   | 10.5             | 90.5                     |
| (2025)DS-YOLO     | RGBD     | 9.2                  | 14.6                   | 10.6             | 90.7                     |
| **BDF-YOLO(ours)**| **RGBD** | **8.5**              | **7.3**                | **6.2**          | **97.0**                 |

## Results on LLVIP
| Method            | Modality | mAP50  | mAP50-95 |
| :---------------: | :------: | :----: | :------: |
| YOLO11            | RGB      | 0.900  |  0.523   |
| (2021)CFT         | RGB+IR   | 0.975  |  0.636   |
| (2023)DIVFusion   | RGB+IR   | 0.898  |  0.520   |
| (2023)CSAA        | RGB+IR   | 0.943  |  0.592   |
| (2024)TFDet       | RGB+IR   | 0.957  |  0.561   |
| (2024)RSDet       | RGB+IR   | 0.958  |  0.613   |
| (2024)AMFD        | RGB+IR   | 0.952  |  0.583   |
| (2024)TEXT-IF     | RGB+IR   | 0.941  |  0.602   |
| (2024)Fusion-Mamba| RGB+IR   | 0.970  |  0.643   |
| (2025)PedDet      | RGB+IR   | 0.958  |  0.568   |
| (2025)Dream-IF    | RGB+IR   | 0.968  |  0.643   |
| (2025)DS-YOLO     | RGB+IR   | 0.970  |  0.653   |
| **BDF-YOLO(ours)**          | **RGB+IR**| **0.976** | **0.673** |



