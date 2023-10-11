# Generating Virtual On-body Accelerometer Data from Virtual Textual Descriptions for Human Activity Recognition

### [Project Page](Coming soon) | [Paper](https://dl.acm.org/doi/10.1145/3594738.3611361) 

The development of robust, generalized models for human activity recognition (HAR) has been hindered by the scarcity of large-scale, labeled data sets.  Recent work has shown that virtual IMU data extracted from videos using computer vision techniques can lead to substantial  performance improvements when training HAR models combined with small portions of real IMU data. Inspired by recent advances in motion synthesis from textual descriptions and connecting Large Language Models (LLMs) to various AI models, we introduce an automated pipeline that first uses ChatGPT to generate diverse textual descriptions of activities. These textual descriptions are then used to generate 3D human motion sequences via a motion synthesis model, T2M-GPT,  and later converted to streams of virtual IMU data. We benchmarked our approach on three HAR datasets (RealWorld, PAMAP2, and USC-HAD) and demonstrate that the use of virtual IMU training data generated using our new approach leads to significantly improved HAR model performance compared to only using real IMU data. Our approach contributes to the growing field of cross-modality transfer methods and illustrate how HAR models can be improved through the generation of virtual training data that do not require any manual effort.

![Example Image](IMUGPT.png)


If our project is helpful for your research, please consider citing :
``` 
@inproceedings{Leng2023generating,
author = {Leng, Zikang and Kwon, Hyeokhyen and Ploetz, Thomas},
title = {Generating Virtual On-Body Accelerometer Data from Virtual Textual Descriptions for Human Activity Recognition},
year = {2023},
doi = {10.1145/3594738.3611361},
booktitle = {Proceedings of the 2023 ACM International Symposium on Wearable Computers},
series = {ISWC '23}
}
```

## 🚩 News

- [2023/07/20] **Paper accepted by UbiComp/ISWC 2023!**
- [2023/07/15] Init project
- [2023/05/04] Uploaded Paper to Arxiv

## More details about IMUGPT coming soon

