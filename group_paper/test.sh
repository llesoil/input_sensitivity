#!bin/bash

vid="$1"
logfilename=$vid.txt
{ time x264 --no-mixed-refs --ref 9 --rc-lookahead 20 --output test.264 $vid.mkv ; } 2> $logfilename
size=`ls -lrt test.264 | awk '{print $5}'`
echo $size
{ time x264  --no-mixed-refs --ref 5 --rc-lookahead 20 --output test.264 $vid.mkv ; } 2> $logfilename
size=`ls -lrt test.264 | awk '{print $5}'`
echo $size
{ time x264 --no-cabac --no-mixed-refs --no-mbtree --ref 5 --rc-lookahead 60 --output test.264 $vid.mkv ; } 2> $logfilename
size=`ls -lrt test.264 | awk '{print $5}'`
echo $size
{ time x264 --no-cabac --no-mixed-refs --no-mbtree --ref 1 --rc-lookahead 60 --output test.264 $vid.mkv ; } 2> $logfilename
size=`ls -lrt test.264 | awk '{print $5}'`
echo $size
{ time x264 --no-cabac --no-mbtree --ref 9 --rc-lookahead 40 --output test.264 $vid.mkv ; } 2> $logfilename
size=`ls -lrt test.264 | awk '{print $5}'`
echo $size
{ time x264 --ref 1 --rc-lookahead 20 --output test.264 $vid.mkv ; } 2> $logfilename
size=`ls -lrt test.264 | awk '{print $5}'`
echo $size
{ time x264 --no-mixed-refs --ref 9 --rc-lookahead 40 --output test.264 $vid.mkv ; } 2> $logfilename
size=`ls -lrt test.264 | awk '{print $5}'`
echo $size
{ time x264 --no-cabac --no-mbtree --ref 5 --rc-lookahead 40 --output test.264 $vid.mkv ; } 2> $logfilename
size=`ls -lrt test.264 | awk '{print $5}'`
echo $size
{ time x264 --no-cabac --no-mixed-refs --no-mbtree --ref 9 --rc-lookahead 60 --output test.264 $vid.mkv ; } 2> $logfilename
size=`ls -lrt test.264 | awk '{print $5}'`
echo $size
{ time x264 --no-cabac --ref 5 --rc-lookahead 40 --output test.264 $vid.mkv ; } 2> $logfilename
size=`ls -lrt test.264 | awk '{print $5}'`
echo $size
