# pairwise-light-cluster

我们的 2D 光照在 1080p 场景里每帧把几百个点光源塞进一个 cluster，GPU 端每个 tile 的 light 列表都超上限，后半屏的灯干脆被丢掉；屏幕边缘的灯因为取整直接消失；透明层和场景层共用同一份 cluster，阴影贴图跟着闪烁；多个相机的 cluster 缓存互相污染；灯数量变化时每帧都重建整张表。render/light_cluster.py 是核心，tests/test_light_cluster.py 15个红。要修的是：按 tile 尺寸计算光源覆盖范围并夹紧屏幕边界；每个 tile 只保留最亮的 N 个灯；按强度和距离排序；剔除完全离屏的灯但保留边缘接触的灯；cluster 结果按相机隔离；静态灯缓存要能复用。

## 运行

```bash
python -m pytest tests -q
```

初始状态：15 个用例失败。修复实现使全部用例通过，不要修改 `tests/` 下的测试。
