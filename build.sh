#!/usr/bin/env sh

set -e
bash ~/.bashrc

projectDir=/home/sgdd/test
test -e "$projectDir/build" && rm -r "$projectDir/build"
mkdir "$projectDir/build"
cd "$projectDir/build"

cmake -DCMAKE_BUILD_TYPE=Debug ..
make

