import os
import json
import pandas as pd

base_dir = "."

rows = []

pch_acc_sum = 0.0
total_num = 0

# 按字典序遍历项目目录
for project in sorted(os.listdir(base_dir)):
    project_path = os.path.join(base_dir, project)

    if not os.path.isdir(project_path):
        continue

    commits_path = os.path.join(project_path, "commits")

    # 判断是否是项目目录
    if not os.path.isdir(commits_path):
        continue

    # 按字典序遍历commit目录
    for commit in sorted(os.listdir(commits_path)):
        commit_path = os.path.join(commits_path, commit)

        if not os.path.isdir(commit_path):
            continue

        fast_dir = os.path.join(commit_path, "fast.o.iclang")
        json_path = os.path.join(fast_dir, "compile.json")

        if not os.path.isfile(json_path):
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            originalTimeMs = data.get("originalTimeMs")
            makePCHTimeMs = data.get("makePCHTimeMs")
            if makePCHTimeMs == 0:
                makePCHTimeMs = originalTimeMs
            pchTimeMs = data.get("pchTimeMs")
            if pchTimeMs == 0:
                pchTimeMs = originalTimeMs
            pch_acc_sum += 1.0 * originalTimeMs / pchTimeMs
            total_num += 1

            rows.append({
                "项目名": project,
                "commit文件名": commit,
                "originalTimeMs": originalTimeMs,
                "makePCHTimeMs": makePCHTimeMs,
                "pchTimeMs": pchTimeMs,
            })

        except Exception as e:
            print(f"读取失败: {json_path}, {e}")

df = pd.DataFrame(rows, columns=[
    "项目名",
    "commit文件名",
    "originalTimeMs",
    "makePCHTimeMs",
    "pchTimeMs",
])

df.to_excel("output.xlsx", index=False)

print("统计完成，已保存为 output.xlsx")
print(f"pch_acc: {pch_acc_sum / total_num}")