import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
import math
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset, DataLoader


# 神经网络
class NNLess_cost(nn.Module):
    def __init__(self):
        super(NNLess_cost, self).__init__()  # N*2*4096

        self.fc1 = nn.Linear(39, 64)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(64, 16)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        out = self.fc1(x[:, 2:41])
        out = self.relu5(out)
        out = self.fc2(out)
        out = self.relu6(out)
        out = self.fc3(out)

        product = x[:, 0] * x[:, 1]
        out = out.squeeze(1)
        denominator = out + product
        safe_denominator = torch.where(denominator == 0, torch.tensor(1e-10, device=denominator.device), denominator)
        q = product * out / safe_denominator

        return q, out

    # 随机函数


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 数据
class DDataset(Dataset):
    def __init__(self, X_descriptor, Y_property, idxs=None):
        self.X = X_descriptor
        self.Y = Y_property
        if idxs is None: idxs = np.arange(self.X.size()[0])
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i = self.idxs[idx]
        x, y = self.X[i], self.Y[i]
        return x, y


def DDataloader(X_descriptor, Y_property, idxs=None, shuffle=True, batch_size=50):
    dataset = DDataset(X_descriptor=X_descriptor, Y_property=Y_property, idxs=idxs)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataloader


# 训练函数
def trainer(model, train_loader, test_loader, optimizer, loss_func, epochs, sample_count, mb_size, path):
    train_log, test_log = [], []
    r2_best = -math.inf
    MAE_best = 0
    count_batch = int(sample_count / mb_size) + 1
    for epoch in range(epochs):
        # for epoch in tqdm(range(epochs), position=0, leave=True):
        epoch_loss = 0
        model.train()
        for batch_x, batch_y in train_loader:
            batch_y_pre, _ = model(batch_x)
            loss = loss_func(batch_y_pre.squeeze(), batch_y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + (loss / count_batch)  # 计算每个epoch损失
        train_log.append(epoch_loss.item())  # 计算平均损失

        model.eval()
        with torch.no_grad():
            for test_x, test_y in test_loader:
                test_y_pre, _ = model(test_x)
                loss_test = loss_func(test_y_pre.squeeze(), test_y.squeeze())
                MAE = mean_absolute_error(test_y.squeeze(), test_y_pre.squeeze())
                r2 = r2_score(test_y.squeeze(), test_y_pre.squeeze())
                if r2_best < r2:  # 保存r2最优的模型
                    loss_best = loss_test
                    torch.save(model.state_dict(), path + 'best_test_model.pth')
                    MAE_best = MAE
                    r2_best = r2
                print('Iter-{}; Total loss: {:.4}; MAE2: {:.4}; r2_score_v: {:.4}'.format(epoch, loss_test.item(), MAE,
                                                                                          r2))
            test_log.append(torch.mean(loss_test).item())

    return loss_best, MAE_best, r2_best


# 预测函数
def model_prediction(model, descriptor_train, descriptor_test, y_train_real, y_test_real, parent_dire):
    with torch.no_grad():
        y_train_pre, out_1 = model(descriptor_train)
        y_test_pre, out_2 = model(descriptor_test)

        loss_train = loss_func(y_train_pre.squeeze(), y_train_real.squeeze())
        MAE_train = mean_absolute_error(y_train_real.squeeze(), y_train_pre.squeeze())
        r2_train = r2_score(y_train_real.squeeze(), y_train_pre.squeeze())
        loss_test = loss_func(y_test_pre.squeeze(), y_test_real.squeeze())
        MAE_test = mean_absolute_error(y_test_real.squeeze(), y_test_pre.squeeze())
        r2_test = r2_score(y_test_real.squeeze(), y_test_pre.squeeze())

        plt.scatter(y_train_real, y_train_pre, c='none', marker='o', edgecolors='g', s=20)
        plt.scatter(y_test_real, y_test_pre, color="b", s=20)
        plt.legend(["train", "test"], fontsize=20)
        plt.plot([torch.min(y_train_real) - 100, torch.max(y_train_real) + 100],
                 [torch.min(y_train_real) - 100, torch.max(y_train_real) + 100], 'r', linewidth=2)
        # plt.yscale("log")
        # plt.xscale("log")
        # plt.xlim(-0.2,20)
        # plt.ylim(-0.2,20)
        plt.tick_params(labelsize=12)
        plt.ylabel('y_pre', {'size': 20})
        plt.xlabel('y_true', {'size': 20})
        plt.savefig(parent_dire + 'test.png')
        plt.show()

    return loss_train, MAE_train, r2_train, loss_test, MAE_test, r2_test, y_train_pre, y_test_pre, y_train_real, y_test_real, out_1, out_2


def out_prediction(model, descriptor):
    with torch.no_grad():
        y_out_pre = model(descriptor)

    return y_out_pre


root_dir = 'E:/物理约束的神经网络丙烯丙烷分离MOF高通量筛选/提交材料/propane/'

label_train = pd.read_csv(root_dir + 'propane.csv', header=0, low_memory=False)
descriptor_train = pd.read_csv(root_dir + 'train_list.csv', header=0, low_memory=False)
label_train_add = pd.read_csv(root_dir + 'propane_add.csv', header=0, low_memory=False)
descriptor_train_add = pd.read_csv(root_dir + 'train_add.csv', header=0, low_memory=False)
data_out = pd.read_csv(root_dir + 'test_opt.csv', header=0, low_memory=False)
label_train.columns = ['propane']
label_train_add.columns = ['propane']
log = []

seed = 248851  # random.randint(0,1000000) #248851
setup_seed(seed)

x_train, x_test, y_train, y_test = train_test_split(
    descriptor_train.iloc[:, 1:42],
    label_train.iloc[:, 0],
    train_size=0.75,
    random_state=seed,
    shuffle=True
)

# 将额外的训练数据全部添加到训练集中
x_train = pd.concat([x_train, descriptor_train_add.iloc[:, 1:42]], axis=0)
y_train = pd.concat([y_train, label_train_add.iloc[:, 0]], axis=0)

# 如果需要合并后的所有数据（可选）
label_all = pd.concat([label_train, label_train_add], axis=0)
descriptor_all = pd.concat([descriptor_train, descriptor_train_add], axis=0)
x_train = torch.from_numpy(x_train.values)
x_test = torch.from_numpy(x_test.values)
y_train = torch.from_numpy(y_train.values)
y_test = torch.from_numpy(y_test.values)

x_train = x_train.to(torch.float32)
y_train = y_train.to(torch.float32)
x_test = x_test.to(torch.float32)
y_test = y_test.to(torch.float32)

input_size = x_train.shape[0]
LR = 0.001
epochs = 1000
mb_size = 50
loss_train_log = []
loss_test_log = []

train_loader = DDataloader(x_train, y_train, idxs=None, shuffle=True, batch_size=mb_size)
test_loader = DDataloader(x_test, y_test, idxs=None, shuffle=False, batch_size=x_test.shape[0])

model = NNLess_cost()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.MSELoss()

loss_best, MAE_best, r2_best = trainer(model, train_loader, test_loader, optimizer, loss_func, epochs, input_size,
                                       mb_size, root_dir)

model_best = NNLess_cost()
model_best.load_state_dict(torch.load(root_dir + 'best_test_model.pth'))

# explainer = shap.DeepExplainer(model_best,x_train)
# shap_values = explainer.shap_values(x_train).reshape(x_train.size(0),x_train.size(1))
# feature_names = ['1.0e-03', '1.0e-02', '1.0e-01', '1.0e+00', '1.0e+01','1.0e+02', '1.0e+03', '2.0e+03', '4.0e+03', '6.0e+03','8.0e+03',
#                      '1.0e+04', '1.2e+04', '1.4e+04', '1.6e+04','1.8e+04', '2.0e+04', '2.4e+04', '2.8e+04', '3.0e+04',
#                      '3.2e+04', '3.6e+04', '4.0e+04', '5.0e+04', '6.0e+04','7.0e+04', '8.0e+04', '9.0e+04', '1.0e+05']
# shap.summary_plot(shap_values,x_train,feature_names=feature_names,show=False)
# plt.savefig(parent_dir+'test'+'_'+str(i)+'.png')
# plt.close()
# i=i+1

loss_train, MAE_train, r2_train, loss_test, MAE_test, r2_test, y_train_pre, y_test_pre, y_train_real, y_test_real, out_1, out_2 = model_prediction(
    model_best, x_train, x_test, y_train, y_test, root_dir)

y_out_pred, out = out_prediction(model_best, torch.from_numpy(data_out.iloc[:, 1:42].values).to(torch.float32))
y_out_pred = pd.DataFrame(y_out_pred.numpy(), columns=['prediction_propane'])
y_out = pd.concat([data_out.iloc[:, 0], y_out_pred], axis=1)
y_out.to_csv(root_dir + 'y_out.csv', header=True, index=False)
out_numpy = out.cpu().numpy()
out_df = pd.DataFrame(out_numpy)
out_df.to_csv(root_dir + 'Out.csv', header=True, index=False)

result_out = {"SEED": seed, "loss_train": loss_train, "MAE_train": MAE_train, "r2_train": r2_train,
              "loss_test": loss_test, "MAE_test": MAE_test, "r2_test": r2_test}
torch.save([y_train_pre, y_test_pre, y_train_real, y_test_real, out_1, out_2], root_dir + 'y_value.pth')
log.append(result_out)

torch.save(log, root_dir + 'log.pth')

