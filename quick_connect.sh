#!/bin/bash
# 快速连接GitHub仓库脚本

echo "=== 快速连接GitHub仓库 ==="
echo ""

# 检查并配置Git用户信息
if [ -z "$(git config --global user.email)" ]; then
    echo "需要配置Git邮箱："
    read -p "请输入你的GitHub邮箱: " GIT_EMAIL
    git config --global user.email "$GIT_EMAIL"
    echo "✅ Git邮箱已配置: $GIT_EMAIL"
else
    echo "✅ Git邮箱已配置: $(git config --global user.email)"
fi

echo ""
echo "Git用户名: $(git config --global user.name)"
echo ""

# 获取仓库名称
read -p "请输入GitHub仓库名称 (默认: Natural-HRI-LLaVA-NeXT-Funturning-Formal): " REPO_NAME
REPO_NAME=${REPO_NAME:-Natural-HRI-LLaVA-NeXT-Funturning-Formal}

REPO_URL="https://github.com/Lesong-Jia/${REPO_NAME}.git"

echo ""
echo "将使用以下仓库URL: $REPO_URL"
read -p "确认创建并连接此仓库? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "已取消"
    exit 0
fi

# 检查远程仓库是否已存在
if git remote get-url origin &>/dev/null; then
    echo "⚠️  远程仓库已存在，正在更新..."
    git remote set-url origin "$REPO_URL"
else
    echo "正在添加远程仓库..."
    git remote add origin "$REPO_URL"
fi

echo "✅ 远程仓库已配置: $REPO_URL"
echo ""
echo "=== 下一步操作 ==="
echo ""
echo "1. 在GitHub上创建仓库: https://github.com/new"
echo "   仓库名: $REPO_NAME"
echo "   不要勾选 'Initialize with README'"
echo ""
echo "2. 创建仓库后，运行以下命令推送代码："
echo ""
echo "   git add ."
echo "   git commit -m 'Initial commit'"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
