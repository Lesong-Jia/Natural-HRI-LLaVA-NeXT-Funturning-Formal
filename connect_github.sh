#!/bin/bash
# 连接本地Git仓库到GitHub的脚本

echo "=== 连接GitHub仓库配置脚本 ==="
echo ""

# 检查是否已配置Git用户信息
if [ -z "$(git config --global user.name)" ]; then
    echo "⚠️  未检测到Git用户配置"
    echo "请先配置Git用户信息："
    echo ""
    read -p "请输入你的GitHub用户名: " GIT_USERNAME
    read -p "请输入你的GitHub邮箱: " GIT_EMAIL
    git config --global user.name "$GIT_USERNAME"
    git config --global user.email "$GIT_EMAIL"
    echo "✅ Git用户信息已配置"
else
    echo "✅ Git用户信息已配置:"
    echo "   用户名: $(git config --global user.name)"
    echo "   邮箱: $(git config --global user.email)"
fi

echo ""
echo "请提供你的GitHub仓库信息："
read -p "GitHub仓库URL (例如: https://github.com/username/repo.git 或 git@github.com:username/repo.git): " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "❌ 仓库URL不能为空"
    exit 1
fi

# 添加远程仓库
echo ""
echo "正在添加远程仓库..."
git remote add origin "$REPO_URL" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ 远程仓库已添加"
elif [ $? -eq 3 ]; then
    echo "⚠️  远程仓库已存在，正在更新..."
    git remote set-url origin "$REPO_URL"
    echo "✅ 远程仓库URL已更新"
else
    echo "❌ 添加远程仓库失败"
    exit 1
fi

echo ""
echo "=== 配置完成 ==="
echo ""
echo "下一步操作："
echo "1. 添加文件: git add ."
echo "2. 提交: git commit -m 'Initial commit'"
echo "3. 推送到GitHub: git push -u origin master"
echo ""
echo "或者如果你想使用main分支："
echo "   git branch -M main"
echo "   git push -u origin main"
