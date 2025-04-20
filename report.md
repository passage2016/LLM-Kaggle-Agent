
# LLM Agent Project Report

## 1. 选题与数据集介绍 / Task and Dataset Introduction
我们选择了 AutoGluon 官方推荐的数据集——California Housing（加州房价数据）。该数据集包含多个影响房价的特征变量，适合用于回归建模任务。

We selected the California Housing dataset, a widely-used benchmark for regression tasks. It includes features like median income, house age, and average occupancy.

## 2. Agent设计思路 / Agent Design Strategy
本项目构建了一个自动化机器学习代理（LLM Agent），主要流程如下：
1. 使用DeepSeek Code模型生成建模代码
2. 自动执行该代码
3. 自动评估结果与性能指标
4. 支持后续提交与可视化

We designed a full pipeline for automated ML via LLM. The agent decomposes tasks, generates AutoGluon code, executes it, and interprets the result. DeepSeek Code API is used for code generation.

## 3. 实验结果与效果 / Results and Performance
模型的RMSE约为 `~0.65`，表现优良，能够稳定预测目标值。

The model achieves RMSE of approximately ~0.65 on the test set, showing promising performance for a fully automated solution.
