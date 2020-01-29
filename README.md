# model_oemof

This repository contains all model development using oemof within the open_MODEX project.

# How to run on Windows OS

Currently the easiest way to run it on Windows OS is to use the Windows Subsystem for Linux. This is an optional feature. Search for 'Turn windows features on or off'. You will get access to a control panel, select 'Windows Subsystem for Linux'.

Then proceed to install the `Ubuntu 18.04 LTS` App from the `Microsoft Store`. Launch the app after installation, you will land in an Ubuntu bash shell.

Type in the following commands:

```
sudo apt-get update
sudo apt-get install python3-pip
sudo apt-get install libglpk-dev python3-swiglpk glpk-utils

git clone https://github.com/open-modex/model_oemof.git
cd model_oemof/
git checkout feature/cross-modelling-exercise

pip3 install -r requirements.txt

python3 oemof.tabular.tutorial.py
```
