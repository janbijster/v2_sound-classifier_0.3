# V2 Sound Classifier v0.3 - Run on raspberry pi

Note: the steps below were taken on a Raspberry Pi 3 Model B+ (Q1 2018, revision number a020d3) 

## 1. Python environment
### 1.1 Berryconda
The easiest way to install all the necessary packages is to install Berryconda. 
We need Python 3 so also Berryconda3. See this [github readme](https://github.com/jjhelmus/berryconda#quick-start) for the right version.

For Raspberry Pi 2 and 3:
(Make sure the rapberry pi has a working internet connection.)
```shell
wget -P ~/Downloads/ "https://github.com/jjhelmus/berryconda/releases/download/v2.0.0/Berryconda3-2.0.0-Linux-armv7l.sh"
~/Downloads/Berryconda3-2.0.0-Linux-armv7l.sh
```

To install, you need to approve the licence terms. Then let berryconda install itself in the default location.

When asked if the install should prepend the Berryconda3 install location to PATH, choose yes. This will make the berryconda python3 the default one.

### 1.2 Python packages
#### 1.2.1 Numba
The most dificult package to install is librosa, because it's dependency numba depends on llvm-lite which will not easily install with pip3. We install it following [this guide](https://numba.pydata.org/numba-doc/dev/user/installing.html#installing-on-linux-armv7-platforms):

```shell
conda install -c numba numba
```

#### 1.2.2 pip
Now we can install the other packages through the standard python package manager *pip*. But to make sure pip finds the conda packages, we first install pip *inside* the conda environment:

```shell
conda install pip
```

Now check if the pip command refers to the right version:

```shell
which pip
```
If the output has `berryconda` in the path (e.g. `/home/pi/berryconda3/bin/pip`) we're good.

#### 1.2.3 other packages
Now we can install the rest of the dependencies:
```shell
pip install librosa Keras==2.2.4 tensorflow==1.13.1 Pillow pydub
```
(It may work with other versions of Keras and tensorflow as well, but these ones are tested.)

## 2. Other tools
For mp3 support we need ffmpeg or libav
```shell
# libav
apt-get install libav-tools libavcodec-extra

####    OR    #####

# ffmpeg
apt-get install ffmpeg libavcodec-extra
```