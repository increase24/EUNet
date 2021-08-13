# EUNet: CNN based Feature Fusion of sEMG and Ultrasound Signals in Hand Gesture Recognition
## introduction
This is an official pytorch implementation of [Feature Fusion of sEMG and Ultrasound Signals in Hand Gesture Recognition](https://ieeexplore.ieee.org/document/9282818/). Please cite this paper if you find this repo helpful for you.

EUNet has two version: one-stream and two stream.

:triangular_flag_on_post: one-stream EUNet is designed for hand gesture recognition based on seperate sEMG or A-mode ultrasound signals;

:triangular_flag_on_post: two-stream EUNet is designed for hand gesture recognition based on fusion sEMG and A-mode ultrasound signals.

+ EUNet(one stream)
![EUNet_oneStream](/figs/EU-Net.png)
The shared CNN architecture for separate sEMG or ultrasound signal to feature extraction and classification.

+ EUNet(two stream)
![EUNet_twoStream](/figs/EUNet_twoStream.png)
The two stream CNN architecture for sEMG and ultrasound feature extraction, feature fusion and classification.


## Environment
The code is developed using python 3.7 on Ubuntu 18.04. NVIDIA GPUs are needed.

## Data preparing
The complete hybrid sEMG/US dataset is not released now. We apply collected sEMG/US data of one subject for code testing, which can be downloaded from: [Baidu Disk](https://pan.baidu.com/s/1qitEFqvwPmD20HnbqgsDcg) 
(code: h99k).


## Usage

### Installation
1. Clone this repo
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a soft link to the dir you save the dataset:
   ```
   ln -s **datadir_save** data
   ```

### Training
* Train EUNet for sEMG modality
  ```
  sh scripts/train_emg_EUNet.sh
  ```
* Train EUNet for A-mode ultrasound modality
  ```
  sh scripts/train_us_EUNet.sh
  ```

### Validate
* Validate EUNet for sEMG modality
  ```
  sh scripts/test_emg_EUNet.sh
  ```
* Validate EUNet for A-mode ultrasound modality
  ```
  sh scripts/test_us_EUNet.sh
  ```

### Contact
If you have any questions, feel free to contact me through Github issues.