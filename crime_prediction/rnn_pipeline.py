#dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
# https://www.kaggle.com/code/devananjelito/predicting-airline-passengers-with-pytorch-rnn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import math


'''
sns.set_style("whitegrid")
path = "international-airline-passengers.csv"
df = pd.read_csv(path) # read file
df.columns = ["month", "passengers"] # change col name
df = df.iloc[:-1] # skip unused rows
df["month"] = pd.to_datetime(df["month"], format="%Y-%m") # cast month to datetime
df = df.dropna(how="all", axis="index") # dropna rows
df = df.sort_values("month")
'''

crime_df = pd.read_csv('Final_Pandas_tensor_2023.csv')
crime_df.columns = ["month", "zipcode", "Homicide_Criminal", "Rape","Robbery_No_Firearm","Aggravated_Assault_No_Firearm","Burglary_Residential",
              "Thefts","Motor_Vehicle_Theft","Other_Assaults", "Arson", "Forgery_and_Counterfeiting", "Fraud", "Embezzlement", "Receiving_Stolen_Property",
              "Vandalism/Criminal_Mischief","Weapon_Violations","Prostitution_and_Commercialized_Vice","Other_Sex_Offenses",
              "Narcotic/Drug_Law_Violations","Gambling_Violations","Offenses_Against_Family_and_Children","DRIVING_UNDER_THE_INFLUENCE",
              "Liquor_Law_Violations", "Public_Drunkenness", "Disorderly_Conduct", "Vagrancy/Loitering", "All_Other_Offenses",
              "psa_1", "psa_2", "psa_3", "psa_4", "psa_A", "total_hours"] # change col name

zipcode_num = 19122
crime_type = 'All_Other_Offenses'
df = crime_df.loc[crime_df['zipcode'] == zipcode_num, ['month', crime_type]]
#print(crime_df.query('zipcode' = zipcode_num))
print(df)


'''
class PassengerDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.sequence_length].astype(np.float32)
        y = self.data[idx + self.sequence_length].astype(np.float32)
        return x, y


passenger_counts = df[crime_type].values
sequence_length = 3  # we will use data of 12 months to predict the passenger in 13th month - need to change
batch_size = 1

dataset = PassengerDataset(passenger_counts, sequence_length)
test_size = 3  # 12 months for test
train_size = len(dataset) - test_size
#print(len(dataset), train_size, test_size)

train_dataset = Subset(dataset, range(0, train_size))
test_dataset = Subset(dataset, range(train_size, len(dataset)))
assert len(train_dataset) + len(test_dataset) == len(dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size)
        self.i2o = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.relu = nn.ReLU()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.relu(hidden)
        output = self.i2o(hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


input_size = sequence_length
output_size = 1  # predict 1 month
hidden_size = 32
rnn = RNN(input_size, hidden_size, output_size)

################
def train(inputs, target):
    hidden = rnn.init_hidden(batch_size)  # initialize hidden
    output, hidden = rnn(inputs, hidden)  # forward pass
    target = target.view(batch_size, output_size)  # resize target to match the output
    loss = torch.sqrt(criterion(output, target))  # compute loss, user sqrt to take RMSE instead of MSE

    # backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def predict(inputs, target, hidden):
    rnn.eval()
    with torch.no_grad():
        output, hidden = rnn(inputs, hidden)  # forward pass
        target = target.view(batch_size, output_size)  # resize target to match the output

        return output, target
################
num_epochs = 100
learning_rate = 0.0002
criterion = nn.MSELoss()
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

print_step = 20
all_losses = []

for epoch in range(num_epochs):

    loss_this_epoch = []

    for inputs, target in train_loader:
        loss = train(inputs, target)
        loss_this_epoch.append(loss.item())

    loss_this_epoch = np.array(loss_this_epoch).mean()
    all_losses.append(loss_this_epoch)
    #if (epoch == 0) or ((epoch + 1) % print_step == 0):
    #    print(f"Epoch {epoch+1: <3}/{num_epochs} | loss = {loss_this_epoch}")

y_true = []
y_pred = []

hidden = rnn.init_hidden(batch_size)
for inputs, target in test_loader:
    output, target = predict(inputs, target, hidden)
    y_pred.append(output.item())
    y_true.append(target.item())

y_true = (np.array(y_true))
y_pred = (np.array(y_pred))
rmse = np.sqrt(np.mean(y_true - y_pred) ** 2)
mae = np.mean(abs(y_true - y_pred))

print(y_true, np.floor(y_pred))

fig, ax = plt.subplots(1,1)
#plt.plot(y_true, label='True Values', marker='o')
plt.plot(y_pred, label='Predicted Values', marker='o')
plt.xlabel('Month')
plt.ylabel('Number of ' + crime_type.replace('_',' '))
plt.title(f'Predicted Number of ' + crime_type.replace('_',' '))
plt.ylim(0, max(max(y_true), max(y_pred))+1)
#plt.xticks([0,1,2])
x_ticks_labels = ['October', 'November', 'December']
ax.set_xticklabels(x_ticks_labels, fontsize=10)
plt.xticks([0,1,2]) #'October', 'November', 'December'
plt.legend()
plt.show()
'''
