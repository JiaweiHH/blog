# 世界线

基于 Hugo 和 Bear Blog 的个人博客。首页显示文章列表，固定使用浅色模式。

## 本地预览

安装 Hugo（当前验证版本：`0.167.0`），在项目目录运行：

```sh
git submodule update --init --recursive
hugo server -D
```

打开 <http://localhost:1313/>。

## 写文章

```sh
hugo new content blog/my-new-post.md
```

编辑 `content/blog/my-new-post.md`。如果文章包含 `draft = true`，正式发布前改为 `false`。

## Cloudflare Pages 发布

在 Cloudflare 的 Workers & Pages 中创建 **Pages** 项目，导入 GitHub 仓库 `JiaweiHH/blog`。

| 配置项 | 值 |
| --- | --- |
| 生产分支 | `main` |
| 框架预设 | `Hugo` |
| 构建命令 | `hugo --minify --baseURL "$CF_PAGES_URL"` |
| 构建输出目录 | `public` |
| 根目录 | 留空（仓库根目录） |
| 环境变量 | `HUGO_VERSION=0.167.0`，同时应用于 Production 和 Preview |

构建命令会使用 Cloudflare 分配的网址覆盖 `hugo.toml` 中的示例 `baseURL`。取得正式域名后，可以将配置中的 `baseURL` 改为该地址；使用自定义域名时，需要同时调整生产环境构建命令中的 `--baseURL`。

首次部署完成后，推送到 `main` 会自动重新发布。无需提交 `public/` 构建产物。

参考：[Cloudflare 官方 Hugo 部署指南](https://developers.cloudflare.com/pages/framework-guides/deploy-a-hugo-site/)。
