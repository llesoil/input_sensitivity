#!/bin/bash

set -xv
extension=".mkv"

cat listVideo.csv | while read line
  do
    read -d, path videoname < <(echo $line)
    newPath=$videoname$extension
    echo $newPath
    x64configs=`ls ./scripts/*.sh`
    csvOutput="./res_ugc/$videoname.csv"
    if test -f "$csvOutput"; then
      echo "$FILE exist"
    else
      cp $path $newPath
      echo "Starting to work with video: " $videoname
      header='configurationID,cabac,ref,deblock,analyse,me,subme,mixed_ref,me_range,trellis,8x8dct,fast_pskip,chroma_qp_offset,bframes,b_pyramid,b_adapt,direct,weightb,open_gop,weightp,scenecut,rc_lookahead,mbtree,qpmax,aq-mode,size,usertime,systemtime,elapsedtime,cpu,frames,fps,kbs'
      touch $csvOutput
      cat /dev/null > $csvOutput
      echo "$header" > $csvOutput
      for x264config in $x64configs
      do
        echo "Processing: " $x264config
        csvLine=`sh $x264config $newPath`
        echo "$csvLine" >> $csvOutput
      done
      rm $newPath
  fi
done

tar cvf "res_ugc.tar.gz" ./res_ugc/*.csv
echo "Done with video" $videoname
