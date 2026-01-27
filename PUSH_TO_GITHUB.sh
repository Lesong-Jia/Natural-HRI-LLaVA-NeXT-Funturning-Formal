#!/bin/bash
# 推送代码到GitHub的脚本

cd /home/lesong_llava/Natural-HRI-LLaVA-NeXT-Funturning-Formal

echo "=== 推送代码到GitHub ==="
echo ""

# 添加所有文件
echo "1. 添加文件到暂存区..."
git add .
echo "✅ 文件已添加"
echo ""

# 创建提交
echo "2. 创建提交..."
git commit -m "Initial commit: Natural-HRI-LLaVA-NeXT fine-tuning project"
echo "✅ 提交已创建"
echo ""

# 切换到main分支（如果当前是master）
echo "3. 切换到main分支..."
git branch -M main 2>/dev/null || echo "已在main分支"
echo ""

# 推送到GitHub
echo "4. 推送到GitHub..."
echo "⚠️  如果是第一次推送，可能需要输入GitHub用户名和Personal Access Token"
echo ""
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 代码已成功推送到GitHub！"
    echo "   仓库地址: https://github.com/Lesong-Jia/Natural-HRI-LLaVA-NeXT-Funturning-Formal"
else
    echo ""
    echo "❌ 推送失败"
    echo "   请确保："
    echo "   1. 已在GitHub上创建了仓库"
    echo "   2. 已配置了GitHub认证（Personal Access Token）"
    echo "   3. 仓库URL正确"
fi
