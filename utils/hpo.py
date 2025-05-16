import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import optuna


class hpoModel(nn.Module):
    def __init__(self, in_features, hidden_units, out_features):
        super().__init__()
        self.dense1=nn.Linear(in_features, hidden_units)
        self.relu=nn.ReLU()
        self.dense2=nn.Linear(hidden_units, out_features)

    def forward(self, X):
        X=self.relu(self.dense1(X))
        X=self.dense2(X) 
        return X
       
class MyDataset(Dataset):
    def __init__(self,x,y):
        super().__init__()
        self.x=x
        self.y=y
        
    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, index):
        return self.x[index],self.y[index] 

x_train=torch.randint(1,10,size=(100,10)).float()
y_train=torch.normal(3,1,(100,1))
train_set=MyDataset(x_train,y_train)
loader=DataLoader(train_set,16,shuffle=True)

#-----------------------------------------------

# 目标函数
def objective(trial):
    '''
    Optuna 目标函数。
    Args:
        trial (optuna.trial.Trial): Optuna 的 Trial 对象，用于建议超参数。
    Returns:
        float: 要优化的指标值 (例如，验证损失或准确率的负数)。
               Optuna 默认会最小化这个返回值。
    '''
    
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    hidden_units = trial.suggest_int("hidden_units", 16, 64, step=4)
    epochs=50
    print(f"配置: LR={lr:.6f}, HU: {hidden_units}, epochs: {epochs}")
    
    model = hpoModel(x_train.shape[-1], hidden_units, 1)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    loss_fn=nn.MSELoss()
    
    model.train()
    for epoch in tqdm(range(epochs),desc="Training"):
        total_loss=0
        num_batches=0
        for x,y in loader:
            optimizer.zero_grad()
            out=model(x)
            l=loss_fn(out,y)
            l.backward()
            optimizer.step()
            total_loss+=l.item()
            num_batches+=1
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        final_loss=avg_loss
        if (epoch+1)%10==0:
            print(f"epochs: {epoch+1} loss: {avg_loss}")
        # report:用于剪枝 (Pruning)
        # 需要定期报告中间值，以便剪枝器判断是否提前终止表现不佳的试验。
        trial.report(avg_loss,epoch)
        
        # should_prune() :检查是否应该剪枝
        if trial.should_prune():
            print(f"  Trial {trial.number} pruned at epoch {epoch+1}.")
            raise optuna.TrialPruned()  # 抛出此异常，Optuna 会标记此试验为 PRUNED
    return final_loss



if __name__=="__main__":

    # 创建一个 Study 对象。Study 对象管理整个优化过程
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource_ratio=0.1, # 表示最低资源量为 max_epochs * 0.1
            reduction_factor=2, # reduction_factor=2 表示减半因子eta=2
            min_early_stopping_rate=0 # 从最低资源等级开始就可能剪枝
            ))

    num_trials = 10 # 想要运行的试验次数
    print(f"Starting optimization with {num_trials} trials...")
    try:
        study.optimize(objective, n_trials=num_trials)
    except KeyboardInterrupt:
        print("Optimization stopped by user.")
    except Exception as e:
        print(f"An error occurred during optimization: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n--- Optimization Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")

    # Optuna API: study.best_trial
    # 返回迄今为止最好的试验对象。
    if study.best_trial:
        best_trial = study.best_trial
        print("\nBest trial:")
        print(f"  Value (minimized loss): {best_trial.value:.4f}") # 最佳试验的目标函数返回值
        print("  Params: ")
        for key, value in best_trial.params.items(): # 最佳试验的超参数
            if isinstance(value, float):
                print(f"    {key}: {value:.6f}")
            else:
                print(f"    {key}: {value}")
    else:
        print("No trials were completed successfully.")

    # Optuna API: study.trials_dataframe()
    # 将所有试验的结果转换为 pandas DataFrame，方便分析。需要安装 pandas。
    try:
        df = study.trials_dataframe()
        print("\nAll trials (DataFrame):")
        print(df)
    except ImportError:
        print("\nInstall pandas to see trials_dataframe: pip install pandas")
    except Exception as e:
        print(f"Could not generate trials_dataframe: {e}")