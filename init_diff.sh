#!/bin/sh

set -e

mv_old_or_new() {
    echo "Mv fast.o.iclang to $1.o.iclang"
    ./run.sh list | while read -r item; do
        echo "Handle $item"
        cd $item/commits
        find . -type d -name "fast.o.iclang" | while read dir; do
            mv "$dir" "$(dirname "$dir")/$1.o.iclang"
        done
        cd ../..
    done
}

# run ./run.sh init all + ./run.sh build all first.
./run.sh iclang-init all SourceRangeCheck
./run.sh change-version all all old
./run.sh fast-build all all --iclang -j 20
mv_old_or_new old
./run.sh change-version all all new
./run.sh fast-build all all --iclang -j 20
mv_old_or_new new