#!/usr/bin/env bash

session_arr=('riiiziiimong/ir_ph1_v2/10')
checkpoint_arr=(282)

for (( i = 0 ; i < ${#session_arr[@]} ; i++ )) ; do
    echo "submit ${session_arr[$i]} ${checkpoint_arr[$i]}"
    ../nsml submit "${session_arr[$i]} ${checkpoint_arr[$i]}"
    sleep 1h
    sleep 5m
done
