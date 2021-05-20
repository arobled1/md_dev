#!/bin/bash
#===================================================================================
# Bash script that uses 'convert' by ImageMagick and 'ffmpeg' to convert a set of .rgb 
#   (snapshot) or .tga (tachyon) files into a movie in mp4 format. ffmpeg line for using 
#   files from tachyon rendering still needs tweaking, especially if a specific image 
#   resolution is desired.
#   - Alan Robledo (edited 5/20/21)
#===================================================================================

# Number of images you want to convert into .png
c=1500
for i in $(seq -f "%05g" 0 $c); do
# If using 'snapsot' from VMD, use line below.
  convert snap.$i.rgb img_$i.png
# If using 'tachyon' from VMD use line below.
 # convert tachyon$i.dat.tga img_$i.png
done
# If using 'snapsot' from VMD, use line below.
ffmpeg -f image2 -framerate 25 -i img_%05d.png -vcodec libx264 -pix_fmt yuv420p -crf 25 video.mp4
# If using 'tachyon' from VMD try using line below. 
# ffmpeg -f image2 -framerate 25 -i img_%05d.png -vcodec libx264 -pix_fmt yuv420p -s 1300x900 -crf 25 video.mp4
