Directory for showing scripts (and resulting video) that turn configurations from an MD simulation into a movie using VMD, ImageMagick, and FFMPEG. 
'upload.vmd' can be run by opening the VMD app and runnning the line:

source upload.vmd

in the Tk Console under 'Extensions'. Then, in your terminal, run the line: 

./make_video.sh

Note: xyz files currently need to be in the same directory as 'upload.vmd' and 'make_video.sh'.
