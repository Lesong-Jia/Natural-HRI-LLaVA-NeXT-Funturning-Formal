# 连接GitHub仓库指南

## 步骤1: 配置Git邮箱（如果还没配置）

```bash
git config --global user.email "你的GitHub邮箱"
```

## 步骤2: 在GitHub上创建仓库

1. 访问 https://github.com/new
2. 仓库名称：`Natural-HRI-LLaVA-NeXT-Funturning-Formal`（或你喜欢的名字）
3. 选择 Public 或 Private
4. **不要勾选** "Initialize this repository with a README"
5. 点击 "Create repository"

## 步骤3: 连接本地仓库到GitHub

创建仓库后，GitHub会显示仓库URL，然后运行：

```bash
# 添加远程仓库（使用HTTPS方式）
git remote add origin https://github.com/Lesong-Jia/Natural-HRI-LLaVA-NeXT-Funturning-Formal.git

# 或者使用SSH方式（如果你配置了SSH密钥）
# git remote add origin git@github.com:Lesong-Jia/Natural-HRI-LLaVA-NeXT-Funturning-Formal.git
```

## 步骤4: 提交并推送代码

```bash
# 添加所有文件
git add .

# 创建第一次提交
git commit -m "Initial commit: Natural-HRI-LLaVA-NeXT fine-tuning project"

# 推送到GitHub（如果是main分支）
git branch -M main
git push -u origin main

# 或者如果是master分支
# git push -u origin master
```

## 注意事项

- 如果使用HTTPS，推送时可能需要输入GitHub用户名和Personal Access Token（不是密码）
- 如果使用SSH，需要先配置SSH密钥
- 如果遇到认证问题，可以查看：https://docs.github.com/en/authentication
