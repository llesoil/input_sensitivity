#!/bin/bash

set -xv

configs=`ls ./scripts/*.sh`
for c in $configs
do
  echo "Processing: $c"
  echo $conf
  bash $c "$conf"
done
