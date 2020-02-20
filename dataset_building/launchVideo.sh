#!/bin/bash

set -xv
videoname="$1"
videofolder="$3"
echo "Starting to work with video: " $videoname
header='configurationID,no_8x8dct,no_asm,no_cabac,no_deblock,no_fast_pskip,no_mbtree,no_mixed_refs,no_weightb,rc_lookahead,ref,size,usertime,systemtime,elapsedtime,cpu,frames,fps,kbs'
x64configs=`ls ./scripts/*.sh`
mkdir -p ./output/$videofolder
for i in `seq 1 $2`;
do
csvOutput="./output/$videofolder/x264-results$i.csv"
touch $csvOutput
cat /dev/null > $csvOutput
echo "$header" > $csvOutput
for x264config in $x64configs
do
echo "Processing: " $x264config
   csvLine=`sh $x264config $videoname`
   echo "$csvLine" >> $csvOutput
done
tar cvf "$videofolder.tar.gz" ./output/$videofolder/x264-results*.csv
#tar cvf "oX264-results$i.tar.gz" /srv/local/macher/bench3/output/*.log
done
echo "Done with video" $videoname
