#!/bin/bash
# 快速开始脚本

set -e

echo "=================================================="
echo "知识蒸馏项目 - 快速开始"
echo "=================================================="

# 检查是否在正确目录
if [ ! -f "train_distillation.py" ]; then
    echo "❌ 错误：请在distillation目录下运行此脚本"
    exit 1
fi

echo ""
echo "📦 步骤1: 检查环境"
echo "--------------------------------------------------"

# 检查Python环境
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "❌ 未找到Python！请先安装Python 3.8+"
    exit 1
fi

echo "✅ Python: $($PYTHON_CMD --version)"

# 检查依赖
echo ""
echo "检查必需的包..."
MISSING_DEPS=()

for pkg in torch pytorch_lightning pandas numpy; do
    if ! $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
        MISSING_DEPS+=($pkg)
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "❌ 缺少以下依赖: ${MISSING_DEPS[*]}"
    echo ""
    echo "请运行以下命令安装："
    echo "  conda activate <your_env>"
    echo "  pip install torch pytorch-lightning pandas numpy"
    exit 1
fi

echo "✅ 所有依赖已安装"

echo ""
echo "🧪 步骤2: 运行测试"
echo "--------------------------------------------------"

$PYTHON_CMD test_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 测试失败！请检查错误信息"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ 环境验证完成！"
echo "=================================================="

echo ""
echo "📖 下一步："
echo ""
echo "1️⃣  准备数据和Teacher模型"
echo "   确保有以下文件："
echo "   - Teacher权重: ../logs/best_discrete_gestures.pt"
echo "   - 数据目录: /path/to/emg_data/"
echo "   - 划分文件: /path/to/split.csv"
echo ""
echo "2️⃣  开始蒸馏训练"
echo "   $PYTHON_CMD train_distillation.py \\"
echo "       --teacher_checkpoint ../logs/best_discrete_gestures.pt \\"
echo "       --data_dir /path/to/emg_data \\"
echo "       --split_csv /path/to/split.csv \\"
echo "       --output_dir ./student_models"
echo ""
echo "3️⃣  监控训练进度"
echo "   tensorboard --logdir student_models/lightning_logs"
echo ""
