#!/bin/sh
# Author : MOULISHANKAR M R

echo " "
echo "  HAPPY TO HAVE YOU ON INDIAN CURRENCY VALUE PREDICTION ! "
echo " "
echo " Checking the Required Dependencies ! "
echo " "

sudo apt install python3-pip -y 
pip3 install numpy
pip3 install pandas
pip3 install sklearn
pip3 install datetime
pip3 install matplotlib

echo " "
echo " Now, you are all set to execute this program ! "
echo " "

python3 main.py
