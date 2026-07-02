# P0 就绪清单：服务器环境 & 数据集字段核对

> 日期：2026-07-02 · 对应 [work_plan.md](work_plan.md) P0（W1–2）与决策门 **G0** · 服务器细节见 `../metaiot_server_guide.md`
> 图例：☐ 待办 · 🧑 需人工/管理员 · 🤖 可脚本化 · ⚠️ 风险点

---

## A. 服务器环境就绪（WHU MetaIoT）

### A1. 访问
- ☐ 🧑 生成 SSH Key：`ssh-keygen -t rsa -b 4096`，公钥 `~/.ssh/id_rsa.pub` 发管理员（@李春伸）开通权限（**禁止密码登录**）。
- ☐ VS Code 装 **Remote-SSH**，写入 WHU Host（校内 `125.220.157.154:22` / 校外 `whuserver.metaiot.group:44022`，User `metaiot_guest`）。
- ☐ 验证登录 + `atop` 查资源（退出 `Ctrl+B` 再 `D`）；确认可用 GPU 型号/显存/是否共享排队。

### A2. 计算环境
- ☐ Conda（`~/anaconda3`）建环境，命名规范 **`<缩写>_radar`**（如 `mg_radar`），Python 3.10。
- ☐ CUDA 对齐：`which nvcc` → `source ~/commonscript/switch_cuda.sh <版本>`（仅当前终端生效），与 PyTorch 版本匹配。
- ☐ 依赖分组安装并冻结 `environment.yml` / `requirements.txt`：
  - 核心：`torch`(+匹配CUDA)、`numpy`、`einops`、`pyyaml`、`tqdm`
  - 点云/扩散：`open3d`、`chamferdist` 或 `pytorch3d`、`diffusers`、`spconv`(匹配CUDA)
  - 下游检测：**OpenPCDet** 或 **mmdetection3d**（VoD CenterPoint / VoxelNeXt 基线，对标 4D-RaDiff / RadarGen）
  - 评估/日志：`scipy`（MMD/JSD）、`wandb` 或 `tensorboard`
- ☐ 🤖 环境冒烟测试：跑通一次极小扩散 step + Chamfer/MMD 单测，确认无 CUDA/算子缺失。

### A3. 存储与工程
- ☐ 目录：数据集 `~/data/public_dataset/{vod,truckscenes}`；高频缓存走 NVMe `~/data_cache`，软链回工作区
  `ln -s ~/data_cache/radar/vod ~/Workspace/radar_gen/data/vod`
- ☐ GitHub 代理：`setproxy`（用完 `unsetproxy`），clone 本项目 repo 到 `~/Workspace`。
- ☐ 大文件传输用 `croc send/收`（数据/权重）。
- ☐ 🧑 ACL 共享给协作者（勿 `chmod 777`）：
  `setfacl -R -m u:<协作者>:rwx ~/Workspace/radar_gen && setfacl -R -d -m u:<协作者>:rwx ~/Workspace/radar_gen`
- ☐ 工程骨架：`configs/ data/ models/ losses/ eval/ scripts/ results/`；实验固定随机种子、config 化。

### A4. 就绪产出
- ☐ 一页《环境就绪清单》（GPU/CUDA/torch 版本、env 名、目录、协作权限）钉在 repo Wiki 或 issue。

---

## B. 数据集就绪 & 字段核对

### B1. 获取与许可
- ☐ 🧑 **View-of-Delft (VoD)**：intelligent-vehicles.org 申请学术下载许可 → 放 `~/data/public_dataset/vod`，装 VoD devkit。
- ☐ 🧑 **MAN TruckScenes**：官方申请/下载（NeurIPS 2024）→ `~/data/public_dataset/truckscenes`，装 truckscenes-devkit。
- ☐ 校验完整性（帧数、序列数、md5/大小），记录版本。

### B2. 逐点字段核对（**关键**，跑通再进 P1）
> 目标：确认每点属性、Doppler 的 raw/补偿、静/动标注来源、坐标系与 ego-pose。**字段名以各自 devkit 为准（下表标 ⚠️ 者必须读 devkit 复核）。**

| 维度 | View-of-Delft | MAN TruckScenes | 待核实动作 |
|------|---------------|-----------------|-----------|
| 逐点字段 | x,y,z, v_r, RCS | x,y,z, 径向 Doppler, RCS | ⚠️ 打印一帧点云 dtype/列名 |
| Doppler | ✅ **raw + 补偿**都有 | ✅ 径向（⚠️确认 raw/补偿口径） | ⚠️ 核对补偿是否已去自车分量 |
| 自车速度来源 | 里程计 | RTK-GNSS + 双 IMU | 取 `v_ego` 供解析约束 `−v_ego·r̂` |
| RCS/反射 | RCS(dBsm) | RCS | 单位核对 |
| LiDAR 配对 | ✅ 有 | ✅ 有 | LiDAR→Radar 可做 |
| ego-pose/外参 | ✅ | ✅ | 传感器坐标系→车体系变换矩阵 |
| 帧率 | ~10Hz（标注） | ~20Hz（全 sweeps） | B 线时序主用 TruckScenes |
| 静/动标注 | 3D 框 | 3D 框（+track ID?） | 用框判静/动、取 `v_obj` |

### B3. 已知陷阱核实（来自调研，务必复核）
- ⚠️ **VoD 雷达为多扫累积**（`t = 0, −1, −2` 叠加）→ 做「逐帧序列生成」会失真。**处置**：B 线时序主用 **TruckScenes**；VoD 只做**单帧**（主线 A）。读 devkit 确认累积规则。
- ⚠️ **Doppler 口径**：raw（含自车）vs 补偿（去自车）——静态解析约束 `v_r=−v_ego·r̂` 需用 **raw**；一致性损失若用补偿则公式改为 `v_r^comp=(v_obj−v_ego)·r̂ + v_ego·r̂`。**先对齐口径再写损失。**
- ⚠️ **坐标/符号约定**：径向方向 `r̂=p/‖p‖` 的正负、雷达安装外参、v_r 正方向（朝向/远离传感器）——用**静态点实测 v_r 反推 v_ego** 做一次自洽性校验。
- ⚠️ 静/动分割质量：先用 GT 框验证「约束上限」，再评估自动分割噪声的影响（对应风险 R3）。

### B4. 数据统计报告（P0 交付，🤖 脚本产出）
- ☐ 每序列帧数、逐帧点数分布（均值/分位）。
- ☐ Doppler 分布（raw & 补偿）、RCS 分布。
- ☐ 静/动点比例、动态目标类别分布。
- ☐ **自洽性校验图**：静态点 `−v_ego·r̂` 预测 vs 实测 v_r 的残差直方图（验证物理关系与坐标约定）。

---

## C. G0 通过判据（W2 末）
- ☐ 服务器可登录、环境冒烟测试通过、协作权限就绪。
- ☐ VoD & TruckScenes 下载完成、devkit 可读一帧、字段口径已确认。
- ☐ **自洽性校验**：静态点 `v_r ≈ −v_ego·r̂` 残差在合理量级（验证解析硬约束可行）。
- ☐ 竞品重扫结论：空白仍成立（见 [competitor_rescan_2026-07.md](competitor_rescan_2026-07.md)）。
- → 全绿则进入 **P1**；任一红灯则修复或按风险预案调整（如基线自建、时序改 TruckScenes）。

## D. 需人工先行的阻塞项（尽早启动，避免卡关键路径）
1. 🧑 SSH Key 提交管理员开通（A1）——**今天就发**。
2. 🧑 VoD / TruckScenes 许可申请（B1）——审批可能数天，**优先提交**。
3. 🧑 确认可用 GPU 配额与排队策略（A1）——决定实验批量与并行度。
