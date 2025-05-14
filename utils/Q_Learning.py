import torch
from tqdm import tqdm
import random
import numpy as np
# ---定义环境---
GRID_ROWS=4
GRID_COLS=4
NUM_STATES=GRID_ROWS*GRID_COLS
NUM_ACTIONS=4 # 0向上，1向下，2向左，3向右

START_CONDITION=0 # (0,0)
END_CONDTION=NUM_STATES-1 # (3,3)

PENALTY_COORDS=[(1,2),(2,1),(3,0)] # 假设这些是惩罚块
PENALTY_STATES=[r*GRID_COLS+c for r,c in PENALTY_COORDS]

# 奖励
REWARD_GOAL = 10
REWARD_PENALTY = -10
REWARD_WALL = -1
REWARD_STEP = -0.1 # 每走一步的轻微惩罚

# --- Q_Learning参数 ---
LEARNING_RATE = 0.1 
Gamma = 0.9  # 贝尔曼最优方程参数
EPSILON_START = 1.0  # 初始探索率
EPSILON_END = 0.01   # 最终探索率
EPSILON_DECAY_RATE = 0.999 # 探索率衰减速率 ,每次迭代乘以这个值

NUM_EPOCHS = 15000 # 训练的总轮数 
MAX_STEPS_PER_EPOCH = 100 # 每轮最大步数，防止无限循环

q_table=torch.zeros((NUM_STATES,NUM_ACTIONS)) # Q(s,a)状态s采取a动作的价值

def state_to_idx(state_index):
    '''转换状态索引为坐标值'''
    return state_index//GRID_ROWS,state_index%GRID_COLS

def idx_to_state(row,col):
    '''转换坐标到状态索引'''
    if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
        return row * GRID_COLS + col
    return -1 

def next_state_reward(current_state,action):
    '''根据当前状态和动作计算下一状态、奖励和是否结束'''
    row,col=state_to_idx(current_state)
    next_row,next_col=row,col
    if action==0:
        next_row-=1
    elif action==1:
        next_row+=1
    elif action==2:
        next_col-=1
    elif action==3:
        next_col+=1
        
    if not(0<=next_row<GRID_ROWS and 0<=next_col<GRID_COLS): # 撞墙
        next_state=current_state
        reward=REWARD_WALL
        done=False # 判定是否结束
    else:
        next_state=idx_to_state(next_row,next_col)
        if next_state==END_CONDTION:
            reward=REWARD_GOAL
            done=True
        elif next_state in PENALTY_STATES:
            reward=REWARD_PENALTY
            done=False
        else:
            reward=REWARD_STEP
            done=False
    return next_state,reward,done

def choose_actions(state,epsilon):
    '''ε-greedy 策略选择'''
    if random.random()<epsilon:
        return random.randint(0,NUM_ACTIONS-1)
    else:
        return torch.argmax(q_table[state]).item()
    
def train_agent(epochs,max_steps_per_epoch,epsilon):
    '''训练智能体'''
    print("start training agent...")
    rewards_pre_epoch=[]
    epsilon=EPSILON_START

    for i in tqdm(range(epochs),desc="Training Agent"):
        current_state=START_CONDITION
        total_reward=0
        done=False
        
        for step in range(max_steps_per_epoch):
            action=choose_actions(current_state,epsilon)
            next_state,reward,done=next_state_reward(current_state,action)
            total_reward+=reward
    
            # 下一状态若终止，未来预期奖励为0，同时无未来奖励
            if done:
                traget=reward # 对 Q(s,a) 理想的、更新后的估计值
            else:
                # 选择能使 Q(s', a') 最大的那个动作 a'
                next_reward=torch.max(q_table[next_state])
                # 贝尔曼最优方程，对未来期望的价值折扣，让agent更倾向快速回报
                traget=reward+Gamma*next_reward
                
            old_q_value=q_table[current_state,action]        
            error=traget-old_q_value # 基于实际估计与预期估计之差
            new_q_value=old_q_value+LEARNING_RATE*error
            q_table[current_state,action]=new_q_value
            
            current_state=next_state
            if done:
                break
        rewards_pre_epoch.append(total_reward)
        
        # 衰减探索率
        if epsilon>EPSILON_END:
            epsilon*=EPSILON_DECAY_RATE
    print("Training finished")

def test_agent(q_table):
    '''测试智能体'''
    current_state=START_CONDITION
    path=[(state_to_idx(current_state))]
    steps=0
    total_reward=0
    
    grid_display=[[" - " for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    for r,c in PENALTY_COORDS:
        grid_display[r][c]=' P '
    goal_r,goal_c=state_to_idx(END_CONDTION)
    grid_display[goal_r][goal_c]=' G '
    grid_display[0][0]=' * ' # 初始点直接标记为通过
    action_map={0:"GO UP",1:"GO DOWN",2:"GO LEFT",3:"GO RIGHT"}
    
    print(f"starting from ({state_to_idx(current_state)})")
    # 测试中只选择最优结果
    while current_state!=END_CONDTION and steps<MAX_STEPS_PER_EPOCH:
        action=torch.argmax(q_table[current_state]).item()
        print(f"state: {state_to_idx(current_state)} action: {action_map[action]}")
        
        next_state,reward,done=next_state_reward(current_state,action)
        current_r,current_c=state_to_idx(next_state)
        if grid_display[current_r][current_c]==" - ":
            grid_display[current_r][current_c]=" * "
        
        path.append((state_to_idx(current_state)))
        total_reward+=reward
        current_state=next_state
        steps+=1
        
        if done:
            print(f"reach the goal when step={steps}")
            break
    print(f"path: {path}")
    print(f"total reward{total_reward}")
    for i in grid_display:
        print(" ".join(i))
        
    print(f"q_table: {q_table}")
train_agent(1500,MAX_STEPS_PER_EPOCH,EPSILON_START)
test_agent(q_table)

