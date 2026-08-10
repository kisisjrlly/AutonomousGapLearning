# 硬件故障证据包（Hardware Fault Evidence）

> 生成时间：2026-08-10 21:48 CST
> 用途：提交 Intel / 主板/整机售后（RMA）的故障记录。

## 1. 机器与配置
- CPU：Intel Core i9-13900KF（Raptor Lake，13代 K 系列——Intel 官方承认的 Vmin-shift 退化问题覆盖型号）
- 主板：ASUS PRIME Z790-P WIFI，BIOS 1836（2026-04-16，含微码 0x12B+；Intel Default Settings 已载入，RAPL 确认 PL1=PL2=253W）
- GPU：NVIDIA RTX 4080 16GB，驱动 580.173.02，当前功耗上限 250W（默认 320W）
- 内存：DDR5（XMP 已关闭，JEDEC 默认频率）
- 系统：Ubuntu，内核 6.8.0-136

## 2. 故障现象
- **GPU 满载运行数分钟 → 整机硬冻结**：屏幕卡死、键盘/鼠标无响应、无法 SSH，只能断电重启；
- **死机时零内核日志**：journalctl 在死机点戛然而止，无 MCE、无 GPU Xid、无 soft lockup 记录
  （内核在冻结瞬间已无法写日志，符合硅片/电源硬锁特征）；
- 在 GPU 训练负载下尤其高频（2026-07-26 至 08-10 期间 ≥10 次硬重启）；
- 历史存在内核 soft lockup 记录：CPU#5 卡死 121938 秒（~34 小时），RCU stall，
  见 2026-07-18 journalctl（boot 之前）——一颗核心停止响应核间中断。

## 3. 硬重启时间线（last -x reboot 记录）
reboot   system boot  6.8.0-136-generi Mon Aug 10 17:56   still running
reboot   system boot  6.8.0-136-generi Mon Aug 10 17:48   still running
reboot   system boot  6.8.0-136-generi Mon Aug 10 11:40   still running
reboot   system boot  6.8.0-136-generi Mon Aug 10 11:05   still running
reboot   system boot  6.8.0-136-generi Mon Aug 10 09:34 - 11:01  (01:26)
reboot   system boot  6.8.0-136-generi Mon Aug 10 09:20 - 11:01  (01:41)
reboot   system boot  6.8.0-136-generi Sun Aug  9 16:51 - 11:01  (18:09)
reboot   system boot  6.8.0-136-generi Sun Aug  9 16:37 - 11:01  (18:23)
reboot   system boot  6.8.0-136-generi Sun Aug  9 16:29 - 11:01  (18:32)
reboot   system boot  6.8.0-124-generi Sun Aug  9 15:17 - 11:01  (19:44)
reboot   system boot  6.8.0-136-generi Sun Aug  9 11:46 - 15:16  (03:30)
reboot   system boot  6.8.0-136-generi Sun Aug  9 11:37 - 15:16  (03:39)
reboot   system boot  6.8.0-136-generi Sun Aug  9 11:15 - 15:16  (04:01)
reboot   system boot  6.8.0-136-generi Sun Aug  9 09:44 - 15:16  (05:32)
reboot   system boot  6.8.0-124-generi Mon Jul 27 11:50 - 15:16 (13+03:25)

## 4. 已尝试的缓解（均不足以根治）
- BIOS 升级至 1836 + Intel Default Settings（微码 0x12B+ 阻止继续退化，但无法修复已退化硅片）；
- XMP 关闭（内存 JEDEC 频率）；
- GPU 功耗上限 320W→250W（降低电源瞬态，仍偶发死机）；
- 训练线程数限制、断点续跑（软件侧容错，非根治）。

## 5. 结论与请求
- 症状与 Intel 公布的 13/14代 K 系列 Vmin-shift 退化（负载死机+随机段错误+渐进恶化）高度吻合；
- 依据 Intel 对 13/14 代 K 系列延长至 5 年质保的政策，申请 CPU RMA（换新）；
- 若需排除电源，建议同步检测/更换 PSU（当前总功耗在 CPU253W+GPU250W 联合负载下接近极限）。
