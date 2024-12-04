#!/bin/bash

echo -e ">>> Yuwei Yin - Compute Canada Server Info <<<\n"

# Linux info
echo -e "\n >>> uname -a"
uname -a
echo -e "\n >>> lsb_release -a"
lsb_release -a
echo -e "\n >>> cat /proc/version"
cat /proc/version
echo -e "\n >>> free -h"
free -h
echo -e "\n >>> df -h"
df -h

# Directory
echo -e "\n >>> pwd"
pwd
echo -e "\n >>> ls"
ls
echo -e "\n echo >>> HOME"
echo "${HOME}"

# GPU
#echo -e "\n nvidia-smi"
#nvidia-smi
