# Video Generation

本仓库跟踪 DiffSynth-Studio 上游源码，保存 Wan 训练、推理链路分析，并在上游目录之外承载本地扩展。

- `docs/WAN_COMMIT_ANALYSIS.md`：对应源码版本的模型加载、数据处理、训练、推理及运行边界说明。
- `DiffSynth-Studio/`：上游 submodule，当前锁定 commit `60b709c4dab8eeabce06ee9bc168328ed4c780e8`；允许同步上游版本，但不做本地源码定制。
- `AGENTS.md`：上游同步、本地扩展和文档维护约束。

本地功能、实验、配置和运行产物放在 `DiffSynth-Studio/` 之外。外层仓库保存 submodule 的 commit 指针，不复制跟踪上游的每个源码文件。

## 获取锁定版本

```bash
git clone --recurse-submodules https://github.com/GitMonkey0/video_generation.git
```

已有外层仓库时，在根目录初始化或恢复其锁定的上游版本：

```bash
git submodule update --init --recursive
```

## 同步上游

先确认 submodule 工作区没有本地改动。需要升级时，获取 `.gitmodules` 配置的上游 main 分支版本：

```bash
git submodule update --init --remote --checkout DiffSynth-Studio
git -C DiffSynth-Studio rev-parse HEAD
git diff --submodule=log -- DiffSynth-Studio
```

检查扩展兼容性、更新 README 的版本标识并复核受影响的分析文档，再按需提交外层仓库：

```bash
git add DiffSynth-Studio README.md docs/WAN_COMMIT_ANALYSIS.md
git commit -m "Update DiffSynth-Studio upstream version"
git push
```

只运行 `git pull` 不应视为已升级到上游 main。普通 `git submodule update` 恢复外层记录的 commit，带 `--remote` 的更新才选择上游分支版本；每次升级仍由外层 commit 固定，以便复现与回退。

源码的许可证与署名文件保留在 `DiffSynth-Studio/` 中。
