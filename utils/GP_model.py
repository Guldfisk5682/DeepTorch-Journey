import torch
import gpytorch
import matplotlib.pyplot as plt

# 1. 准备数据
train_x = torch.linspace(0, 10, 50) 
train_y = torch.sin(train_x) + torch.randn(train_x.size()) * 0.2 # 真实函数值加噪声

# 生成一些测试数据点，用于预测
test_x = torch.linspace(0, 10, 100) # 100个测试输入点
# 测试数据没有对应的真实y值，因为我们就是要预测它们

train_x = train_x.float()
train_y = train_y.float()
test_x = test_x.float()

# 2. 定义高斯过程模型
# GPyTorch 要求定义一个继承自 gpytorch.models.ExactGP 的类
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        # likehood:似然函数
        super().__init__(train_x, train_y, likelihood)

        # 定义均值函数：这里我们使用零均值，即在没有数据时，函数取值最可能为0
        self.mean_module = gpytorch.means.ZeroMean()
        # self.mean_module = gpytorch.means.ConstantMean() 也可以使用常数均值

        # 定义协方差函数（核函数）：这里使用 RBF 核
        # ScaleKernel 用于引入一个可学习的振幅（输出尺度）参数
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    # 定义前向传播过程，用于计算给定输入上的均值向量和协方差矩阵
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) # 返回一个表示多元高斯分布的对象

# 3. 初始化似然函数和模型
# 对于回归问题，我们使用高斯似然，它包含了可学习的噪声方差参数
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# 创建模型实例
model = ExactGPModel(train_x, train_y, likelihood)

# 4. 学习模型超参数（通过最大化边缘似然）

# 将模型和似然设置为训练模式
model.train()
likelihood.train()

# 定义优化器，优化模型的参数（核函数超参数和噪声方差）
optimizer = torch.optim.Adam(model.parameters(), lr=0.1) # 可以调整学习率

# 定义损失函数：负对数边缘似然 (Negative Log Marginal Likelihood)
# 最大化边缘似然等价于最小化负边缘似然
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# 开始训练循环
training_iter = 50 # 迭代次数
print("开始训练...")
for i in range(training_iter):
    # 清零之前的梯度
    optimizer.zero_grad()
    # 通过模型计算训练输入上的输出分布
    output = model(train_x)
    # 计算损失 (负边缘似然)
    loss = -mll(output, train_y)
    # 反向传播计算梯度
    loss.backward()

    # 打印一些信息，看看超参数是否在变化
    if i % 10 == 0:
        print(f'迭代 {i+1}/{training_iter} - 损失: {loss.item():.3f} '
              f'长度尺度: {model.covar_module.base_kernel.lengthscale.item():.3f} '
              f'噪声方差: {model.likelihood.noise.item():.3f}')

    # 更新模型参数
    optimizer.step()

print("训练完成。")
print(f"最终长度尺度: {model.covar_module.base_kernel.lengthscale.item():.3f}")
print(f"最终噪声方差: {model.likelihood.noise.item():.3f}")
print(f"最终输出尺度 (振幅^2): {model.covar_module.outputscale.item():.3f}")


# 5. 进行预测

# 将模型和似然设置为评估模式
model.eval()
likelihood.eval()

# 在测试输入上进行预测
# with torch.no_grad() 确保不计算梯度，节省内存和计算
with torch.no_grad():
    # 通过模型得到测试输入上的高斯过程预测分布 (潜在函数值 f(x*) 的分布)
    # model(test_x) 返回的是 gpytorch.distributions.MultivariateNormal 对象
    # 这个分布的均值是后验均值，方差是后验方差（函数值的不确定性）
    # predicted_prior_dist = model(test_x) # 这是对潜在函数 f(x*) 的预测

    # 通过似然函数得到对观测值 y(x*) 的预测分布
    # likelihood(model(test_x)) 返回的也是 gpytorch.distributions.MultivariateNormal 对象
    # 这个分布的均值是 y(x*) 的预测均值 (通常与 f(x*) 相同)，方差是 y(x*) 的预测方差 (包含函数不确定性 + 噪声)
    observed_pred = likelihood(model(test_x))

    # 获取预测均值和预测方差
    # 对于多元高斯分布对象，.mean 属性是均值向量，.variance 属性是方差向量 (对角线元素)
    mean_pred = observed_pred.mean
    variance_pred = observed_pred.variance
    stddev_pred = torch.sqrt(variance_pred) # 预测标准差

    # 构建置信区间 (例如 95% 置信区间，大约是均值加减两倍标准差)
    lower_bound = mean_pred - 2 * stddev_pred
    upper_bound = mean_pred + 2 * stddev_pred


# 6. 可视化结果

plt.figure(figsize=(10, 6))

# 绘制训练数据点
plt.plot(train_x.numpy(), train_y.numpy(), 'o', label='training data')

# 绘制预测均值曲线
plt.plot(test_x.numpy(), mean_pred.numpy(), '-', label='prediction mean', color='blue')

# 绘制不确定性区域 (例如 95% 置信区间)
# 使用 fill_between 函数填充均值上下某个范围
plt.fill_between(test_x.numpy(), lower_bound.numpy(), upper_bound.numpy(), color='blue', alpha=0.2, label='95% Confidence Interval')

plt.xlabel("input X")
plt.ylabel("output Y")
plt.title("GP Regression with PyTorch and GPyTorch")
plt.legend()
plt.grid(True)
plt.show()