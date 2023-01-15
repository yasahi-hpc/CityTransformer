#!/bin/bash

mkdir dataset

modes=(
    "train"
    "val"
    "test"
)

for mode in "${modes[@]}" ; do
    echo "Extract ${mode}.tar.gz"
    cat ${mode}.tar.gz* | tar zxvf -
    mv ${mode} dataset
done
