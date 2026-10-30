# 世界线

基于 Hugo 和 Bear Blog 的个人博客

## 本地预览

安装 Hugo（当前验证版本：`0.167.0`），在项目目录运行：

```sh
git submodule update --init --recursive
hugo server -D
```

打开 <http://localhost:1313/>

## 写文章

```sh
hugo new content blog/my-new-post.md
```

编辑 `content/blog/my-new-post.md`。如果文章包含 `draft = true`，正式发布前改为 `false`

参考：[Hugo 官方 GitHub Pages 部署指南](https://gohugo.io/host-and-deploy/host-on-github-pages/)
