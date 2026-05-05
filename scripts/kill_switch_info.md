# Kill Switch

자동 iteration 루프를 즉시 멈추려면:
```bash
touch /home/ajy/AU-RegionFormer/scripts/kill_switch
```

이 파일이 존재하면 Claude는 현재 iteration 끝난 직후 중단하고 상태 보고.

재개:
```bash
rm /home/ajy/AU-RegionFormer/scripts/kill_switch
```
