# OpenCV_Object_Detection_Automation_with_Actuator

![IMG_2032](https://user-images.githubusercontent.com/52450051/161061541-6e34c0f6-9347-4fb7-bdcd-564c351b35a8.jpg)

## Assignment

We've seen how to learn an object and then detect it, so now we're going to use this to automate an action.

* You learn to recognize an object by creating a its cascade file (xml).
* You configure your raspberry pi (Jetson Nano) to read in a camera feed.
* This object can then be observed with the raspberry pi (Jetson Nano), for this you use openCV
* You add an object counter that displays in the upper left corner how many of these objects it observes.
* When this object is observed you make sure that the raspberry pi (Jetson Nano) controls an actuator (led, relay, lcd, ...)

## Needed Components

![IMG_2034](https://user-images.githubusercontent.com/52450051/161063948-55903995-8caa-43b8-bb0d-7ac78a562a31.jpg)
![IMG_2035](https://user-images.githubusercontent.com/52450051/161064323-237b0921-bcdd-48ff-81ab-4ff9807fe3c2.jpg)

* Jetson Nano 2GB or better, i strongly recommend to buy a 4GB version (Jetson Nano B01) as you will need all memory you can get for AI projects. I used the 2GB version (Jetson Nano 2 GByte), more info can be found here >> https://developer.nvidia.com/embedded/jetson-developer-kits

<img width="847" alt="image" src="https://user-images.githubusercontent.com/52450051/161066329-18a368dc-d028-4153-a457-a76bf582badb.png">

* ICE TOWER for cooling (recommended), see following link for more details >> https://www.jetsonhacks.com/2019/11/30/jetson-nano-extreme-cooling/

![IMG_2037](https://user-images.githubusercontent.com/52450051/161069978-d029efba-f224-4253-acb8-a239335d0024.jpg)

* Fast Micro SD Card, 64GB UHS-1 or higher recommended.
* Power supply and a jumper (J48) for power selection, see following links for details >> https://desertbot.io/blog/jetson-nano-power-supply-barrel-vs-micro-usb

![image](https://user-images.githubusercontent.com/52450051/161082332-7b3e4d69-b637-4c7e-9a8d-a4e22b8cd814.png)

* Keyboard, mouse and monitor, it is also possible to controll the Jetson Nano headless using SSH and/or VNC Viewer.
* Raspberry Pi cam, see following link for cam specifications >> https://www.raspberrypi.com/documentation/accessories/camera.html

![IMG_2036](https://user-images.githubusercontent.com/52450051/161068150-056b9ff8-3616-4d9b-a935-5e1c08d1273e.jpg)

* Optional 3D printed stand for the Pi Cam, STL files are included in this repository.

![IMG_2038](https://user-images.githubusercontent.com/52450051/161073550-af22b952-975f-4e33-b1fe-aea75fcc57dd.jpg)

* 1 LED, 2 resistors, a transistor, a breadbord and some dupont-wires, more details later. 

## Setting up the Jetson Nano

The first thing we need to do is flash a 64GB or larger micro SD card with a special image that can be found folowing this link >> https://github.com/Qengineering/Jetson-Nano-image/tree/611693cfe4aba56e33987afa9690f75259c545c1
The advantage is that this image has several pre-installed frameworks which we will need as well as CUDA support:

* JetPack 4.6.0
* OpenCV 4.5.3 (includes openCV-python and openCV-contrib-python libraries)
* TensorFLow 2.4.1
* TensorFlow Addons 0.13.0-dev
* Pytorch 1.8.1
* TorchVision 0.9.1
* LibTorch 1.8.1
* ncnn 20210720
* MNN 1.2.1
* JTOP 3.1.1
* TeamViewer aarch64 15.24.5

### Expand your image using GPARTED when using SD cards larger than 32GB

```
sudo apt-get install gparted
```
Patition >> Resize/Move >> select the partition

![image](https://user-images.githubusercontent.com/52450051/161077274-abd1427e-257b-4063-b8de-37350e38b0d0.png)

Reboot the device
```
sudo reboot
```

Verify your partition size
```
df -h
```
![image](https://user-images.githubusercontent.com/52450051/161078160-ad614ce3-d4ca-424d-a77d-75660c05cab7.png)

### Run jtop and verify the following settings
```
sudo jtop
```
![image](https://user-images.githubusercontent.com/52450051/161078796-2307c85f-ccdd-4e69-9c60-d77c46334f27.png)
![image](https://user-images.githubusercontent.com/52450051/161078905-cc516900-16ab-43bb-b6e0-86ede535b85b.png)

### Enlarge the memory swap

Follow the instructions in this link, i did not erase the swap afterwards >>> https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html

Verify the memory and swap file after reboot.

```
free -m
```

![image](https://user-images.githubusercontent.com/52450051/161080375-60948366-fe1f-48c5-8ff8-bdafcab85b5e.png)

### Install Pycharm for Coding (optional)

This is a very good and reliable IDE with virtual environments (VENV) and auto-completion.
Follow the steps below for installation:

```
wget https://download.jetbrains.com/python/pycharm-community-2019.1.2.tar.gz
tar xvzf pycharm-community-2019.1.2.tar.gz
pycharm-community-2019.1.2/bin/pycharm.sh
```

![image](https://user-images.githubusercontent.com/52450051/161083622-463c2479-6efa-4c74-8922-e79ee6d09006.png)

## Create the HAAR Cascade for AI training

<img width="744" alt="image" src="https://user-images.githubusercontent.com/52450051/161088639-1400990f-ddd4-416c-bf66-07102bf443d1.png">

You will need a Windows machine for this with lots of computing power !!!!

This is an xml file that will generated by training your machine using thousands of positive pictures (cats in this case) and 3 times as much negative pictures (things related to cats, like dogs ....), the more pictures you provide, the more accuracy you will get. Install the training application and Follow the steps using this link >> https://amin-ahmadi.com/cascade-trainer-gui/














