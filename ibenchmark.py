import argparse
import os
import subprocess
from contextlib import contextmanager
from typing import Callable
import shlex
import shutil
import time
from functools import wraps
from enum import Enum
import json
from dataclasses import dataclass
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor
import difflib


# ==================== Config Begin ====================


class TestType(Enum):
    CTEST = 1
    MAKE_CHECK = 2
    OTHER = 3


class ProjectInfo:
    def __init__(self, name: str, url: str, filename: str):
        self.name = name
        self.url = url
        self.filename = filename

        self.init_commands: list[str] = []

        self.basic_cmake_args = "-DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_C_COMPILER=/usr/local/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++"
        self.extra_cmake_args = ""
        self.cmake_file = "../src"

        self.parallel = 64

        self.test_data_url = ""
        self.test_data_filename = ""

        self.test_type = TestType.CTEST
        self.ctest_e = ""
        self.ctest_r = ""
        self.other_test_commands: list[str] = []
        self.test_env : dict[str, str] = {}


project_infos: dict[str, ProjectInfo] = {}


def init_project_infos():
    cwd = os.path.abspath(os.getcwd())

    abseil_cpp: ProjectInfo = ProjectInfo("abseil-cpp",
                                          "https://github.com/abseil/abseil-cpp/archive/e4c43850ad008b362b53622cb3c88fd915d8f714.zip",
                                          "abseil-cpp-e4c43850ad008b362b53622cb3c88fd915d8f714")
    abseil_cpp.init_commands.append("rm -rf Testing")
    abseil_cpp.extra_cmake_args = "-DABSL_BUILD_TESTING=ON -DABSL_USE_EXTERNAL_GOOGLETEST=ON"
    abseil_cpp.ctest_e = "absl_mutex_test|absl_btree_test|absl_log_format_test"

    catch2: ProjectInfo = ProjectInfo("catch2",
                                      "https://github.com/catchorg/Catch2/archive/4c8671cfbbf0019d3827305d4d1c82a74eeb029a.zip",
                                      "Catch2-4c8671cfbbf0019d3827305d4d1c82a74eeb029a")
    catch2.extra_cmake_args = "-DCATCH_DEVELOPMENT_BUILD=ON"
    catch2.ctest_e = "ApprovalTests"

    cvc5: ProjectInfo = ProjectInfo("cvc5",
                                    "https://github.com/cvc5/cvc5/archive/ae82eb306143ade54a6f99b2aae0b62b8c77cd35.zip",
                                    "cvc5-ae82eb306143ade54a6f99b2aae0b62b8c77cd35")
    cvc5.test_type = TestType.MAKE_CHECK

    llvm: ProjectInfo = ProjectInfo("llvm",
                                    "https://github.com/llvm/llvm-project/archive/90c326b198080c5c208f62f6755d54d7b69b291d.zip",
                                    "llvm-project-90c326b198080c5c208f62f6755d54d7b69b291d")
    llvm.extra_cmake_args = "-DLLVM_ENABLE_PROJECTS=\"clang\" -DLLVM_TARGETS_TO_BUILD=X86 -DCLANG_TOOLING_BUILD_AST_INTROSPECTION=OFF"
    llvm.cmake_file = "../src/llvm"
    llvm.test_type = TestType.MAKE_CHECK

    mariadb: ProjectInfo = ProjectInfo("mariadb",
                                       "https://github.com/MariaDB/server/archive/00a9afb5818433c26537ccaf6b2c59ad493dd473.zip",
                                       "server-00a9afb5818433c26537ccaf6b2c59ad493dd473")
    mariadb.init_commands.append("unzip -q dependencies/libmariadb.zip -d src")
    mariadb.init_commands.append("unzip -q dependencies/wsrep-lib.zip -d src")
    mariadb.init_commands.append("unzip -q dependencies/wolfssl.zip -d src/extra/wolfssl/")
    mariadb.init_commands.append("unzip -q dependencies/rocksdb.zip -d src/storage/rocksdb/")
    mariadb.extra_cmake_args = "-DMYSQL_MAINTAINER_MODE=NO -DWITH_LIBFMT=system -DLIBFMT_INCLUDE_DIR=/usr/local/include"

    ncnn: ProjectInfo = ProjectInfo("ncnn",
                                    "https://github.com/Tencent/ncnn/archive/8ccf720530c89d1a03be4e8401013cdfef4bda0b.zip",
                                    "ncnn-8ccf720530c89d1a03be4e8401013cdfef4bda0b")
    ncnn.extra_cmake_args = "-DNCNN_BUILD_TESTS=ON"
    ncnn.ctest_e = "test_squeezenet"

    opencv: ProjectInfo = ProjectInfo("opencv",
                                      "https://github.com/opencv/opencv/archive/b5d38ea4cbfdb911155ed674b7a535839bc3d6f8.zip",
                                      "opencv-b5d38ea4cbfdb911155ed674b7a535839bc3d6f8")
    opencv.init_commands.append("cp -r dependencies src/dependencies")
    opencv.extra_cmake_args = ("-DOPENCV_DOWNLOAD_PATH=" + os.path.join(cwd, "opencv", "src", "dependencies")
                               + " -DOPENCV_SKIP_DOWNLOAD=ON -DBUILD_TESTS=ON")
    opencv.test_data_url = "https://github.com/opencv/opencv_extra/archive/8926a3906a44733d68485c703de1c8a765577246.zip"
    opencv.test_data_filename = "opencv_extra-8926a3906a44733d68485c703de1c8a765577246"
    opencv.ctest_r = ("opencv_test_flann|opencv_sanity_videoio|opencv_perf_videoio|opencv_sanity_photo|"
                      "opencv_sanity_calib3d|opencv_sanity_objdetect|opencv_test_stitching|opencv_sanity_imgcodecs|"
                      "opencv_sanity_dnn|opencv_sanity_features2d|opencv_test_objdetect|opencv_test_features2d|"
                      "opencv_test_ml|opencv_sanity_stitching|opencv_perf_photo")
    opencv.test_env["OPENCV_TEST_DATA_PATH"] = os.path.join(cwd, "opencv", "test", "testdata")

    rocksdb: ProjectInfo = ProjectInfo("rocksdb",
                                       "https://github.com/facebook/rocksdb/archive/72c38871673f34ef22c66b8fcb9292812272a961.zip",
                                       "rocksdb-72c38871673f34ef22c66b8fcb9292812272a961")

    tesseract: ProjectInfo = ProjectInfo("tesseract",
                                         "https://github.com/tesseract-ocr/tesseract/archive/ade0dfaa8cc1b12341286aa91e11f8ab77a035ad.zip",
                                         "tesseract-ade0dfaa8cc1b12341286aa91e11f8ab77a035ad")
    tesseract.init_commands.append("unzip -q dependencies/test.zip -d src")
    tesseract.init_commands.append("unzip -q dependencies/googletest.zip -d src/unittest/third_party/")
    tesseract.extra_cmake_args = "-DBUILD_TESTS=ON"
    tesseract.ctest_e = ("lstm_recode_test|lstm_squashed_test|lstm_test|apiexample_test|baseapi_test|"
                         "baseapi_thread_test|equationdetect_test|lang_model_test|layout_test|loadlang_test|"
                         "lstmtrainer_test|mastertrainer_test|osd_test|pagesegmode_test|progress_test|"
                         "resultiterator_test|textlineprojection_test|unicharcompress_test")

    z3: ProjectInfo = ProjectInfo("z3",
                                  "https://github.com/Z3Prover/z3/archive/bdb9106f996c838f387c1602f29e9665feccea2f.zip",
                                  "z3-bdb9106f996c838f387c1602f29e9665feccea2f")
    z3.test_data_url = "https://github.com/Z3Prover/z3test/archive/75a38c7d0ae28a979421605af7e377ca7eccfaef.zip"
    z3.test_data_filename = "z3test-75a38c7d0ae28a979421605af7e377ca7eccfaef"
    z3.test_type = TestType.OTHER
    z3.other_test_commands.append(
        "python3 " + os.path.join(cwd, "z3", "test", "scripts", "test_benchmarks.py") + " "
        + "./z3" + " " + os.path.join(cwd, "z3", "test", "regressions", "smt2") + " smt2 ${TESTJ}")

    project_infos[abseil_cpp.name] = abseil_cpp
    project_infos[catch2.name] = catch2
    project_infos[cvc5.name] = cvc5
    project_infos[llvm.name] = llvm
    project_infos[mariadb.name] = mariadb
    project_infos[ncnn.name] = ncnn
    project_infos[opencv.name] = opencv
    project_infos[rocksdb.name] = rocksdb
    project_infos[tesseract.name] = tesseract
    project_infos[z3.name] = z3

    for info in project_infos.values():
        os.makedirs(info.name, exist_ok=True)


@dataclass
class CommitInfo:
    source: str
    commit: str
    pch_line: int

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CommitInfo":
        return CommitInfo(
            source=data["source"],
            commit=data["commit"],
            pch_line=data["pchLine"]
        )


def load_commit_info(json_path: str) -> CommitInfo:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CommitInfo.from_dict(data)


# ==================== Config End ====================


# ==================== Utils Begin ====================


@contextmanager
def cd(path):
    old = os.getcwd()
    print(f"cd {path}")
    os.chdir(path)
    try:
        yield
    finally:
        print("cd ..")
        os.chdir(old)


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time: {end - start:.6f} seconds")
        return result

    return wrapper


def run_command(command_args, new_env : dict[str, str] | None = None):
    try:
        env = os.environ.copy()
        if new_env is not None:
            for key, value in new_env.items():
                env[key] = value
        subprocess.run(command_args, check=True, env=env)

    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error: command {shlex.join(command_args)} failed with exit code {e.returncode}")

    except Exception as e:
        raise ValueError(f"Error: failed to execute command {shlex.join(command_args)}: {e}")


def download_with_md5_check(url: str, filepath: str):
    md5path = filepath + ".md5"
    if not os.path.exists(md5path):
        raise ValueError(f"{md5path} not found, fatal error")
    try:
        run_command(["md5sum", "-c", md5path])
        print(f"{filepath} already exists and md5 check passed, skipping")
        return
    except ValueError:
        pass
    print(f"Downloading {filepath} ...")
    try:
        run_command(["wget", url, "-O", filepath])
    except ValueError:
        os.remove(filepath)
        raise
    print(f"Download {filepath} done")


def files_equal(path1: str, path2: str) -> bool:
    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        return f1.read() == f2.read()


def diff_line_numbers(old_path: str, new_path: str):
    with open(old_path, "r", encoding="utf-8") as f:
        old_lines = f.read().splitlines()

    with open(new_path, "r", encoding="utf-8") as f:
        new_lines = f.read().splitlines()

    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines)

    deleted_lines = []
    added_lines = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "delete":
            deleted_lines.extend(range(i1 + 1, i2 + 1))

        elif tag == "insert":
            added_lines.extend(range(j1 + 1, j2 + 1))

        elif tag == "replace":
            deleted_lines.extend(range(i1 + 1, i2 + 1))
            added_lines.extend(range(j1 + 1, j2 + 1))

    return deleted_lines, added_lines


# True: new, False: old
def new_to_bool(content: str) -> bool:
    is_new = True
    if content == "old":
        is_new = False
    elif content == "new":
        is_new = True
    else:
        raise ValueError(f"Invalid value {content} for old-or-new, only support 'old' or 'new'")
    return is_new

# ==================== Utils End ====================


# ==================== Cmd Internal Begin ====================


class GlobalState:
    def __init__(self):
        self.total_task_num = 0
        self.failed_task_num = 0
        self.gMap: dict[str,  Any] = {}


global_state : GlobalState = GlobalState()


def download_project(project_info: ProjectInfo, args: argparse.Namespace):
    download_with_md5_check(project_info.url, "src.zip")
    if project_info.test_data_url != "":
        download_with_md5_check(project_info.test_data_url, "test.zip")


def init_project(project_info: ProjectInfo, args: argparse.Namespace):
    print("Init", project_info.name)
    shutil.rmtree(project_info.filename, ignore_errors=True)
    shutil.rmtree("src", ignore_errors=True)
    run_command(["unzip", "-q", "src.zip"])
    shutil.move(project_info.filename, "src")
    if project_info.test_data_filename != "":
        shutil.rmtree(project_info.test_data_filename, ignore_errors=True)
        shutil.rmtree("test", ignore_errors=True)
        run_command(["unzip", "-q", "test.zip"])
        shutil.move(project_info.test_data_filename, "test")
    for command in project_info.init_commands:
        run_command(shlex.split(command))

    cmake_args = "cmake " + project_info.basic_cmake_args + " " + project_info.extra_cmake_args + " " + project_info.cmake_file
    shutil.rmtree("build", ignore_errors=True)
    os.mkdir("build")
    with cd("build"):
        print(cmake_args)
        run_command(shlex.split(cmake_args))


def iclang_init_project(project_info: ProjectInfo, args: argparse.Namespace):
    run_command(shlex.split("find ./commits -type f -name '*.o' -delete"))
    run_command(shlex.split("find ./commits -type f -name '*.tmp' -delete"))
    run_command(shlex.split("find ./commits -type d -name '*.iclang' -exec rm -rf {} +"))
    run_command(shlex.split("find ./commits -type d -name '*.iclangtmp' -exec rm -rf {} +"))

    iclang_mode = args.mode
    print("IClang init", project_info.name, ":", iclang_mode)
    summary_white_list = []
    summary_pch_info = []
    for commit_name in sorted(os.listdir("commits")):
        commit_dir = os.path.join("commits", commit_name)
        if os.path.isdir(commit_dir):
            commit_info = load_commit_info(os.path.join(commit_dir, "info.json"))
            abs_src_path = os.path.abspath(os.path.join("src", commit_info.source))
            summary_white_list.append(abs_src_path)
            summary_pch_info.append({
                "srcPath": abs_src_path,
                "pchLine": commit_info.pch_line
            })

    if args.skippch:
        summary_pch_info = []
    summary_json_data = {
        "iClangMode": iclang_mode,
        "whiteList": summary_white_list,
        "pchInfo": summary_pch_info
    }
    summary_json_path = os.path.join("commits", "config.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json_data, f, indent=2, ensure_ascii=False)


def change_version_project(project_info: ProjectInfo, args: argparse.Namespace):
    commits_list = []
    if args.commit_name == "all":
        for commit_name in sorted(os.listdir("commits")):
            commit_dir = os.path.join("commits", commit_name)
            if not os.path.isdir(commit_dir):
                continue
            commits_list.append(commit_dir)
    else:
        commits_list.append(os.path.join("commits", args.commit_name))

    is_new = new_to_bool(args.old_or_new)
    is_strict = args.strict

    for commit_dir in commits_list:
        info_json_path = os.path.join(commit_dir, "info.json")
        commit_info = load_commit_info(info_json_path)
        src_path = os.path.join("src", commit_info.source)
        old_cpp_path = os.path.join(commit_dir, "old.cpp")
        new_cpp_path = os.path.join(commit_dir, "new.cpp")

        global_state.total_task_num += 1

        # Check + patch
        # Do not use copy2, use copyfile, discard metadata!
        if not is_new: # old
            if not files_equal(src_path, new_cpp_path):
                if is_strict:
                    raise ValueError(f"{src_path} != {new_cpp_path}")
                else:
                    global_state.failed_task_num += 1
                    print(f"{src_path} is already old")
            else:
                print(f"Replace {src_path} with {old_cpp_path}")
                shutil.copyfile(old_cpp_path, src_path)
        else: # new
            if not files_equal(src_path, old_cpp_path):
                if is_strict:
                    raise ValueError(f"{src_path} != {old_cpp_path}")
                else:
                    global_state.failed_task_num += 1
                    print(f"{src_path} is already new")
            else:
                print(f"Replace {src_path} with {new_cpp_path}")
                shutil.copyfile(new_cpp_path, src_path)


def build_project(project_info: ProjectInfo, args: argparse.Namespace):
    iclang_env = {}
    if args.iclang:
        iclang_env["ICLANG"] = os.path.abspath(os.path.join("commits", "config.json"))

    make_j = project_info.parallel
    if args.j:
        make_j = args.j
    make_args = "make -j " + str(make_j)
    with cd("build"):
        print(make_args)
        run_command(shlex.split(make_args), iclang_env)


def test_project(project_info: ProjectInfo, args: argparse.Namespace):
    test_j = project_info.parallel
    if args.j:
        test_j = args.j
    with cd("build"):
        if project_info.test_type == TestType.CTEST:
            ctest_args = "ctest -j " + str(test_j)
            if project_info.ctest_e != "":
                ctest_args += " -E \"" + project_info.ctest_e + "\""
            if project_info.ctest_r != "":
                ctest_args += " -R \"" + project_info.ctest_r + "\""
            print(ctest_args)
            run_command(shlex.split(ctest_args), project_info.test_env)
        elif project_info.test_type == TestType.MAKE_CHECK:
            make_check_args = "make check -j " + str(test_j)
            print(make_check_args)
            run_command(shlex.split(make_check_args), project_info.test_env)
        else:
            for command in project_info.other_test_commands:
                command = command.replace("${TESTJ}", str(test_j))
                print(command)
                run_command(shlex.split(command), project_info.test_env)


def load_compile_file_command_map(compile_commands_json_path: str):
    if not os.path.exists(compile_commands_json_path):
        raise ValueError(f"Error: JSON file does not exist: {compile_commands_json_path}")

    try:
        with open(compile_commands_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error: Failed to read or parse JSON file: {e}")

    if not isinstance(data, list):
        raise ValueError(f"Error: JSON root must be a list.")

    result = {}

    try:
        for entry in data:
            file_path = entry["file"]
            command = entry["command"]

            result[file_path] = command
    except KeyError as e:
        raise ValueError(f"Error: Missing required field in JSON entry: {e}")

    return result


def fast_build_project_commit(project_info: ProjectInfo, commit_dir: str,
                        compile_file_command_map : dict[str, str], args: argparse.Namespace):
    iclang_env = {}
    if args.iclang:
        iclang_env["ICLANG"] = os.path.abspath(os.path.join("commits", "config.json"))

    info_json_path = os.path.join(commit_dir, "info.json")
    commit_info = load_commit_info(info_json_path)
    src_abs_path = os.path.abspath(os.path.join("src", commit_info.source))
    compile_command = compile_file_command_map[src_abs_path]
    command_args = shlex.split(compile_command)

    # Update output.
    output_path = os.path.join(commit_dir, "fast.o")
    original_output_idx = -1
    for i, item in enumerate(command_args):
        if item == "-o":
            original_output_idx = i + 1
            break
    if original_output_idx == -1 or original_output_idx >= len(command_args):
        raise ValueError("Error: Can not find valid -o output_path in the compile command")
    command_args[original_output_idx] = output_path

    print(f"Fast compile project: {project_info.name}, commit: {commit_dir}, iclang_env: {iclang_env}, command: {shlex.join(command_args)}")

    run_command(command_args, iclang_env)

    print(f"Compile {project_info.name} {commit_dir} Done")


def fast_build_project_commits(project_info: ProjectInfo, args: argparse.Namespace):
    compile_j = 1
    if args.j:
        compile_j = args.j
    futures = []
    executor = ThreadPoolExecutor(max_workers=compile_j)
    compile_commands_json_path = os.path.join("build", "compile_commands.json")
    compile_file_command_map = load_compile_file_command_map(compile_commands_json_path)
    if args.commit_name == "all":
        for commit_name in sorted(os.listdir("commits")):
            commit_dir = os.path.join("commits", commit_name)
            if os.path.isdir(commit_dir):
                futures.append(executor.submit(fast_build_project_commit, project_info, commit_dir,
                                               compile_file_command_map, args))
    else:
        futures.append(
            executor.submit(fast_build_project_commit, project_info, os.path.join("commits", args.commit_name),
                            compile_file_command_map, args))

    for f in futures:
        f.result()

    return


def mark_line_info(line_infos: list[str], src_lines: list[str], desc: str,
                   start_line: int, start_column: int, end_line: int, end_column: int, src_path: str):
    # line granularity
    if start_column-1 > 0 and not src_lines[start_line-1][:start_column-1].isspace():
        start_line += 1
    if end_column < len(src_lines[end_line-1]) and not src_lines[end_line-1][end_column:].isspace():
        end_line -= 1
    for i in range(start_line-1, end_line):
        line_info = line_infos[i]
        if line_info == "other" or (line_info == "class" and "function" in desc):
            line_infos[i] = desc
        else:
            raise ValueError(f"Conflict line info in {os.path.abspath(src_path)}: line {i+1}, prev: {line_infos[i]}, cur: {desc}")


def get_line_infos(src_path: str, compile_json_path: str) -> list[str]:
    with open(src_path, 'r', encoding='utf-8') as f:
        src_lines = f.readlines()

    # Start from 0.
    line_infos : list[str] = ["other"] * (len(src_lines))

    with open(compile_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for idx, item in enumerate(data['declInfos']):
        decl_type = item['type']
        start_line = item['startLine']
        start_column = item['startColumn']
        end_line = item['endLine']
        end_column = item['endColumn']
        if decl_type == 'function' or decl_type == 'class' or decl_type == 'template':
            if decl_type == 'function':
                # expand mangled name flag and tags
                if item['mangledName'] == "":
                    mark_line_info(line_infos, src_lines, "function"+item["tags"]+"(unmangled)", start_line, start_column, end_line,
                                   end_column, src_path)
                else:
                    mark_line_info(line_infos, src_lines, "function"+item["tags"], start_line, start_column, end_line,
                                   end_column, src_path)
            else:
                mark_line_info(line_infos, src_lines, decl_type, start_line, start_column, end_line, end_column,
                               src_path)
        else:
            raise ValueError(f"Invalid decl type: {decl_type}")

    # Mark space and comment
    for i, content in enumerate(src_lines):
        if line_infos[i] != "other":
            continue
        fmt_content = content.strip()
        if fmt_content == "":
            line_infos[i] = "space"
        elif fmt_content[0] == '/':
            # only support //
            line_infos[i] = "comment"

    # for i, content in enumerate(line_infos):
    #     print(i+1, content)

    return line_infos


def sr_sta_project_commit(project_info: ProjectInfo, commit_dir: str, args: argparse.Namespace):
    cpp_path = os.path.join(commit_dir, "new.cpp")
    compile_json_path = os.path.join(commit_dir, "fast.o.iclang", "compile.json")
    line_infos = get_line_infos(cpp_path, compile_json_path)
    for info in line_infos:
        global_state.gMap[info] = global_state.gMap.get(info, 0) + 1


def diff_project_commit(project_info: ProjectInfo, commit_dir: str, args: argparse.Namespace):
    old_cpp_path = os.path.join(commit_dir, "old.cpp")
    new_cpp_path = os.path.join(commit_dir, "new.cpp")
    old_compile_json_path = os.path.join(commit_dir, "old.o.iclang", "compile.json")
    new_compile_json_path = os.path.join(commit_dir, "new.o.iclang", "compile.json")
    deleted_lines, added_lines = diff_line_numbers(old_cpp_path, new_cpp_path)
    old_line_infos = get_line_infos(old_cpp_path, old_compile_json_path)
    new_line_infos = get_line_infos(new_cpp_path, new_compile_json_path)
    # key: desc, value[0]: deleted_num, value[1]: added_num
    res: dict[str, tuple[int, int]] = {}
    for line in deleted_lines:
        key = old_line_infos[line - 1]
        old = res.get(key, (0, 0))
        res[key] = (old[0] + 1, old[1])
    for line in added_lines:
        key = new_line_infos[line - 1]
        old = res.get(key, (0, 0))
        res[key] = (old[0], old[1] + 1)
    global_state.gMap[os.path.abspath(commit_dir)] = res


# ==================== Cmd Internal End ====================


# ==================== Cmd Broadcast Begin ====================

def handle_project_commit(project_name: str, handle_funcs: list[Callable[[ProjectInfo, str, argparse.Namespace], None]],
                   args: argparse.Namespace):
    project_list = []
    if project_name == "all":
        for info in project_infos.values():
            project_list.append(info)
    else:
        if project_name not in project_infos:
            raise ValueError(f"Error: project '{project_name}' not found in ibenchmark")
        project_list.append(project_infos[project_name])

    for info in project_list:
        with cd(info.name):
            commit_list = []
            if args.commit_name == "all":
                for commit_name in sorted(os.listdir("commits")):
                    commit_dir = os.path.join("commits", commit_name)
                    if os.path.isdir(commit_dir):
                        commit_list.append(commit_dir)
            else:
                commit_list.append(os.path.join("commits", args.commit_name))
            for commit_dir in commit_list:
                for handle_func in handle_funcs:
                    handle_func(info, commit_dir, args)


def handle_project(project_name: str, handle_funcs: list[Callable[[ProjectInfo, argparse.Namespace], None]],
                   args: argparse.Namespace):
    project_list = []
    if project_name == "all":
        for info in project_infos.values():
            project_list.append(info)
    else:
        if project_name not in project_infos:
            raise ValueError(f"Error: project '{project_name}' not found in ibenchmark")
        project_list.append(project_infos[project_name])

    for info in project_list:
        with cd(info.name):
            for handle_func in handle_funcs:
                handle_func(info, args)


# ==================== Cmd Broadcast End ====================


# ==================== Cmd Begin ====================


def cmd_list(args):
    for info in project_infos.values():
        print(info.name)


@timeit
def cmd_download(args):
    handle_project(args.project, [download_project], args)


@timeit
def cmd_init(args):
    handle_project(args.project, [init_project], args)


@timeit
def cmd_iclang_init(args):
    handle_project(args.project, [iclang_init_project], args)


def cmd_change_version(args):
    handle_project(args.project, [change_version_project], args)
    print("Total commits num:", global_state.total_task_num)
    print("Skip commits num:", global_state.failed_task_num)


@timeit
def cmd_build(args):
    handle_project(args.project, [build_project], args)


@timeit
def cmd_test(args):
    handle_project(args.project, [test_project], args)


@timeit
def cmd_fast_build(args):
    handle_project(args.project, [fast_build_project_commits], args)


@timeit
def cmd_sr_sta(args):
    handle_project_commit(args.project, [sr_sta_project_commit], args)
    total_num = 0
    for num in global_state.gMap.values():
        total_num += num
    print(f"[total] {total_num}")
    for desc, num in global_state.gMap.items():
        print(f"[{desc}] {num} ({1.0*num / total_num})")


@timeit
def cmd_diff_commits(args):
    handle_project_commit(args.project, [diff_project_commit], args)
    # desc, (deleted num, added num, commit num)
    res: dict[str, tuple[int, int, int]] = {}
    valid_num = 0
    for prefix, diff_map in global_state.gMap.items():
        print(prefix, end="")
        is_valid = True
        for key, value in diff_map.items():
            print(f" [{key}: -{value[0]}, +{value[1]}]", end="")
            old = res.get(key, (0, 0, 0))
            res[key] = (old[0] + value[0], old[1] + value[1], old[2] + 1)

            if not(key == "space" or ("function" in key and "unmangled" not in key and
                                      "auto" not in key and "constexpr" not in key and
                                      "special-attr" not in key and
                                      "default" not in key and
                                      "deleted" not in key and
                                      "invalid-compound-stmt" not in key)):
                is_valid = False
        print()
        if is_valid:
            valid_num += 1


    print("==================== Summary ====================")
    for key, value in res.items():
        print(f"[{key}] -:{value[0]}, +:{value[1]}, commits:{value[2]}")
    print(f"Valid commits: {valid_num}")
    print("========================================")


# ==================== Cmd End ====================


def main():
    parser = argparse.ArgumentParser(prog="ibenchmark", description="IBenchmark manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_parser = subparsers.add_parser("list", help="List all projects in IBenchmark")
    list_parser.set_defaults(func=cmd_list)

    # download
    download_parser = subparsers.add_parser("download", help="Download project source code")
    download_parser.add_argument("project", help="Project under list or 'all'")
    download_parser.set_defaults(func=cmd_download)

    # init
    init_parser = subparsers.add_parser("init", help="rm -rf src, unzip src.zip to src, "
                                                     "init test directory if exists, run other init commands, "
                                                     "init build directory, CMake config project")
    init_parser.add_argument("project", help="Project under list or 'all'")
    init_parser.set_defaults(func=cmd_init)

    # iclang-init
    iclang_init_parser = subparsers.add_parser("iclang-init", help="gen summary config.json under project/commits "
                                                                   "according to IClang mode and project/commits/*/info.json"
                                                                   "rm -rf *.o, *.tmp, *.iclang, *.iclangtmp "
                                                                   "under project/commits/*/.")
    iclang_init_parser.add_argument("project", help="Project under list or 'all'")
    iclang_init_parser.add_argument("mode", help="IClang mode")
    iclang_init_parser.add_argument("--skippch", action="store_true", help="disable provided pch, test auto pch preamble")
    iclang_init_parser.set_defaults(func=cmd_iclang_init)

    # change-version
    change_version_parser = subparsers.add_parser("change-version",
                                                 help="Replace src with commits/xx/old.cpp or commits/xx/new.cpp."
                                                      "For old.cpp, the corresponding src should be equal to new,"
                                                      "For new.cpp, the corresponding src should be equal to old. "
                                                      "If --strict is enabled, we will throw an exception when "
                                                      "encountering a mismatch. Otherwise, we just skip it.")
    change_version_parser.add_argument("project", help="Project under list or 'all'")
    change_version_parser.add_argument("commit_name", help="Commit name under commits (e.g., 01, 02, 03, ...) "
                                                          "under project/commits or 'all'")
    change_version_parser.add_argument("old_or_new", help="'old' / 'new'")
    change_version_parser.add_argument("--strict", action="store_true", help="When check failed, "
                                                                            "throw an exception instead of skipping")
    change_version_parser.set_defaults(func=cmd_change_version)

    # build
    build_parser = subparsers.add_parser("build", help="Build project")
    build_parser.add_argument("project", help="Project under list or 'all'")
    build_parser.add_argument("-j", type=int, help="Parallel jobs")
    build_parser.add_argument("--iclang", action="store_true", help="Enable IClang, "
                                                                             "set env ICLANG=(abs)commits/config.json")
    build_parser.set_defaults(func=cmd_build)

    # test
    test_parser = subparsers.add_parser("test", help="Test project")
    test_parser.add_argument("project", help="Project under list or 'all'")
    test_parser.add_argument("-j", type=int, help="Parallel jobs")
    test_parser.set_defaults(func=cmd_test)

    # fast-build
    fast_build_parser = subparsers.add_parser("fast-build",
                                                 help="Do not build project, compile src directly to commits/commit_name/fast.o "
                                                      "according to compile_commands.json.")
    fast_build_parser.add_argument("project", help="Project under list or 'all'")
    fast_build_parser.add_argument("commit_name", help="Commit name under commits (e.g., 01, 02, 03, ...) "
                                                        "under project/commits or 'all'")
    fast_build_parser.add_argument("-j", type=int, help="Parallel jobs")
    fast_build_parser.add_argument("--iclang", action="store_true", help="Enable IClang, "
                                                                             "set env ICLANG=(abs)commits/config.json")
    fast_build_parser.set_defaults(func=cmd_fast_build)

    # source range statistic
    sr_sta_parser = subparsers.add_parser("sr-sta",
                                        help="Source range statistic. You should run ./init_sr_sta.sh first.")
    sr_sta_parser.add_argument("project", help="Project under list or 'all'")
    sr_sta_parser.add_argument("commit_name", help="Commit name under commits (e.g., 01, 02, 03, ...) "
                                                 "under project/commits or 'all'")
    sr_sta_parser.set_defaults(func=cmd_sr_sta)

    # diff-commits
    diff_parser = subparsers.add_parser("diff-commits",
                                        help="Diff commits, show changed functions, classes, templates. "
                                             "You should run ./init_diff.sh first.")
    diff_parser.add_argument("project", help="Project under list or 'all'")
    diff_parser.add_argument("commit_name", help="Commit name under commits (e.g., 01, 02, 03, ...) "
                                                          "under project/commits or 'all'")
    diff_parser.set_defaults(func=cmd_diff_commits)

    args = parser.parse_args()

    init_project_infos()
    args.func(args)


if __name__ == "__main__":
    main()
