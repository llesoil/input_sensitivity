#!/usr/bin/env bash
SF="$1"
rm TPC-H.db
cd tpch-dbgen
./dbgen -f -s $SF
cd ..
SCALE_FACTOR=$SF make
