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

## 网站图标

图标源文件是 `assets/images/favicon.webp`，在 `hugo.toml` 的 `params.favicon` 中指定。
替换这张正方形图片即可更新图标。Hugo 会自动生成 16、32、48 像素的 PNG 标签页图标和 180 像素的 Apple 主屏幕图标。
生成文件的名称会随图片内容变化，帮助浏览器加载更新后的图标。

替换后，将 `assets/images/favicon.webp` 提交并推送到 GitHub，等待自动发布完成。

## GitHub Pages 发布

博客地址：<https://jiaweihh.github.io/blog/>

仓库的 **Settings → Pages → Source** 使用 **GitHub Actions**。
`.github/workflows/hugo.yaml` 会在推送到 `main` 后自动构建并发布博客，也可以在 Actions 页面手动运行。

日常更新文章后，在项目目录执行：

```sh
git add content
git commit -m "更新文章"
git push
```

如果还修改了配置、图片或模板，也需要将对应文件加入提交。无需提交 `public/` 构建产物。

发布流程固定使用 Hugo `0.167.0`，自动下载主题子模块，并使用 GitHub Pages 提供的网址构建，确保文章和图片链接包含 `/blog/` 路径。

在 [Actions 页面](https://github.com/JiaweiHH/blog/actions) 查看发布进度。

参考：[Hugo 官方 GitHub Pages 部署指南](https://gohugo.io/host-and-deploy/host-on-github-pages/)。
