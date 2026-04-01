#!/bin/sh

set -e

# run ./run.sh init all + ./run.sh build all first.
./run.sh iclang-init all BasicFuncXCheck
./run.sh change-version all all new
./run.sh fast-build all all --iclang -j 20