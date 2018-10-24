#!/usr/bin/env sh

set -e
bash ~/.bashrc

projectDir=/home/sgdd/Optimization-under-Constraint
test -e "$projectDir/build" && rm -r "$projectDir/build"
mkdir "$projectDir/build"
cd "$projectDir/build"

cmake -DCMAKE_BUILD_TYPE=Debug ..
make

