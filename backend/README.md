# FIC 后端模块

## 项目结构

```
├── models      算法模型
├── params      预训练参数
├── public      公共的静态文件、生成文件
├── test        算法模型的测试代码
├── utils       工具函数
├── app.py      服务入口
├── config.py   服务配置
└── dev.sh      以debug模式启动服务脚本（仅linux有效）
```

## 启动服务

- `python app.py` 直接启动服务
- linux 下，可以运行 `dev.sh` 以开发模式启动服务

后端服务默认起在 `1127` 端口，可直接访问 `localhost:1127` 测试连通性。实际情况下，需要经过 Nginx 反向代理至 `/api/` 接口才能够与前端网页对接使用。

## 预训练模型

见 Github releases
