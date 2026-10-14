#!/bin/bash
# 全量夹爪重跑 v2 (2026-10-13): 22 个 session 顺序处理,新管线全部定案参数
cd /home/ss/sroi_dataprocess
for s in /home/ss/data/1000_onesb_labpicking/*-png /home/ss/data/120_onesb_supplementary/*-png; do
  echo "===== $s ====="
  python3 gripper_estimation_april_tag.py "$s" --recursive 2>&1 | grep -E "^---|both=|Run-level|Session gripper|WARNING|no color"
done
echo "ALL_DONE $(date +%H:%M)"
