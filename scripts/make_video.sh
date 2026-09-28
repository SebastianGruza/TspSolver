#!/bin/bash
# assembles ~/santa/frames/*.png (history h*.png + hourly t*.png, by name) into ~/santa/www/santa.mp4 (2 fps, 1080p)
cd ~/santa/frames && ls h*.png t*.png 2>/dev/null | sort > list.txt && sed "s/^/file '/; s/$/'/; s/$/\nduration 0.5/" list.txt > list.ffconcat
ffmpeg -y -loglevel error -f concat -safe 0 -i list.ffconcat -vf "scale=1920:1080,format=yuv420p" -c:v libx264 -r 24 ~/santa/www/santa.mp4 && ls -la ~/santa/www/santa.mp4
