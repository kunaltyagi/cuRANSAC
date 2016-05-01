#! /usr/bin/env sh

string=$HOME/local/opencv3

if [ -z `echo $LD_LIBRARY_PATH | sed 's/:/\n/g' | grep $string` ]; then
    export LD_LIBRARY_PATH=$string:$LD_LIBRARY_PATH;
    echo "Setup Complete";
else
    echo "Setup Already Done";
fi
