# ibenchmark

Env: Ubuntu-22.04 docker container.

## Before testing

(1) Install necessary tools.

```
apt intall -y cmake build-essential lld git wget pkg-config zip unzip diff \
python3-pip openjdk-8-jdk libgmp-dev antlr3 libantlr3c-dev cxxtest \
libgflags-dev libleptonica-dev libicu-dev libpango1.0-dev libcairo2-dev \
libncurses-dev libgnutls28-dev bison tzdata

pip install toml

apt purge libgtest-dev libgmock-dev
rm -rf /usr/src/googletest
git clone -b v1.14.0 https://github.com/google/googletest.git
cd googletest
mkdir build
cd build
cmake ..
make -j 16
make install

wget https://github.com/fmtlib/fmt/releases/download/11.0.2/fmt-11.0.2.zip
unzip -q fmt-11.0.2.zip
rm fmt-11.0.2.zip
cd fmt-11.0.2
mkdir build
cd build
cmake ..
make -j 16
make install
```

(2) Replace GNU ld with lld.

```
rm /usr/bin/ld
ln -s /usr/bin/ld.lld /usr/bin/ld
```

(3) Install IClang.

```
git clone -b illvm-16.0.4 https://github.com/incc-project/illvm.git
cd illvm
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS="clang;compiler-rt" -DLLVM_TARGETS_TO_BUILD="X86" -DCMAKE_BUILD_TYPE=Release ../llvm
make -j 64
make install
```

(4) Download test projects and init.

```
cd ibenchmark
python3 ./ibenchmark.py download all
python3 ./ibenchmark.py init all
```

# Testing

```
cd ibenchmark
python3 ./ibenchmark.py build-commits all all new --re --test --restrict -j 64
# -j 64: parallel jobs, depends on your PC
# Expect no errors and return 0
```

> You can use `python3 ./ibenchmark.py -h` to see more useful commands.

Performance statistics: Todo.