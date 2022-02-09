#!/bin/bash

black Shape_Fun_aLME.py

python Shape_Fun_aLME.py

#ffmpeg -r 12 -i Frame_%1d.png movie1.avi

ffmpeg -framerate 30 -i Frame_%1d.png -vf format=yuv420p movie1.mp4

rm *.png

