# Neural Single-Shot GHz FMCW Correlation Imaging
### [Project Page]() | [Paper]()

[Cindy Pan](), [Noah Walsh](), [Yuxuan Zhang](https://www.alexyuxuanzhang.com/), [Zheng Shi](https://zheng-shi.github.io/), [Felix Heide](https://www.cs.princeton.edu/~fheide/)

If you find our work useful in your research, please cite:
```
@article{HsinPan:24,
author = {Cindy (Hsin) Pan and Noah Walsh and Yuxuan Zhang and Zheng Shi and Felix Heide},
journal = {Opt. Express},
keywords = {Frequency modulation; Imaging techniques; Phase contrast; Real time imaging; Three dimensional imaging; Virtual reality},
number = {16},
pages = {27835--27848},
publisher = {Optica Publishing Group},
title = {Neural single-shot GHz FMCW correlation imaging},
volume = {32},
month = {Jul},
year = {2024},
url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-32-16-27835},
doi = {10.1364/OE.519031},
}
```

This code repository includes code to reproduce the findings of the manuscript. The code will perform the following operations:

1) load FMCW raw signals for real-world scenes captured with the proposed all-optical correlation ToF hardware prototype;
2) decode estimated absolute depths of the pixels of the scene of interest using the frequency decoding network (FDN);
3) perform test-time-optimization and fine-tuning upon the decoded depth using phase measurements extracted from the FMCW raw signals.
4) display the final reconstructed depth and the corresponding surface normals.


## Requirements
This code is developed using Pytorch. Full frozen environment can be found in 'environment.yml', note some libraries in this environment are not necessary to run this code. Please refer to the anaconda official website (https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) for details on how to create an environment using an existing yml file. 

```
conda env create -n single-shot-fmcw -f environment.yml
conda activate single-shot-fmcw
```

## Data
For data of the real-world scenes, their FMCW signals are located in the 'fmcw_signals/' folder available at http://ghz-fmcw.cs.princeton.edu and their phase values are store in the 'phase/' folder. The Synthetic test data presented in the paper are extracted from the Hypersim dataset [Hypersim RGB-D](https://github.com/apple/ml-hypersim). See 'dataloader/' folder for more details. 

## Inference
To perform inference on real-world captures, please download the pre-trained model in the 'network/' folder from http://ghz-fmcw.cs.princeton.edu, then you can run the 'optica_2023.ipynb' notebook in Jupyter Notebook. The notebook will process the measurements and display reconstructed depth.

## License
Our code is licensed under BSL-1. By downloading the software, you agree to the terms of this License. 

## Questions
If there is anything unclear, please feel free to reach out to me at hp0187[at]princeton[dot]edu.
