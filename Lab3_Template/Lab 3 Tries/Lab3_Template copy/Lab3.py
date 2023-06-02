
import matplotlib.pyplot as plt

import torch
import torchvision
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load MNIST Dataset in Numpy

# 1000 training samples where each sample feature is a greyscale image with shape (28, 28)
# 1000 training targets where each target is an integer indicating the true digit
mnist_train_features = np.load('mnist_train_features.npy') 
mnist_train_targets = np.load('mnist_train_targets.npy')

# 100 testing samples + targets
mnist_test_features = np.load('mnist_test_features.npy')
mnist_test_targets = np.load('mnist_test_targets.npy')
y_train=mnist_train_targets
y_test=mnist_test_targets

# Reshape features via flattening the images

train1d = []
for i in range(len(mnist_train_features)):
    train1d.append(np.reshape(mnist_train_features[i],-1))
train1d=np.array(train1d)

test1d = []
for i in range(len(mnist_test_features)):
    test1d.append(np.reshape(mnist_test_features[i],-1))
test1d=np.array(test1d)

print(train1d.shape, test1d.shape)

# Scale the dataset according to standard scaling

train1dS = StandardScaler().fit_transform(train1d)
test1dS = StandardScaler().fit_transform(test1d)

def train_test_split(X,Y,test_size,random_state):
    np.random.seed=random_state
    shuffler = np.random.permutation(Y.shape[0])
    X = X[shuffler]
    Y = Y[shuffler]

    X_train = X[int(test_size*len(Y)):]
    Y_train= Y[int(test_size*len(Y)):]

    X_val = X[:int(test_size*len(Y))]
    Y_val = Y[:int(test_size*len(Y))]

    return X_train, Y_train, X_val, Y_val

# Split training dataset into Train (90%), Validation (10%)

XTrfn, YTrn, XVFn, YVn = train_test_split(train1dS,y_train,test_size=0.2,random_state=1)
XTrf=torch.from_numpy(XTrfn).float()
XVF=torch.from_numpy(XVFn).float()
YTr=torch.from_numpy(YTrn).long()
YV=torch.from_numpy(YVn).long()

class mnistClassification(torch.nn.Module):
        
        def __init__(self, input_dim, hdim1, hdim2, hdim3, output_dim, reg_lambda):
             
            super(mnistClassification, self).__init__()

            self.lay1 = torch.nn.Linear(input_dim, hdim1)
            self.lay2 = torch.nn.Linear(hdim1, hdim2)
            self.do = torch.nn.Dropout(0.15)
            self.lay3 = torch.nn.Linear(hdim2,hdim3)
            self.lay4 = torch.nn.Linear(hdim3, output_dim)
            self.reg_lambda = reg_lambda  # regularization strength
        
        def forward(self, x):

            x1 = torch.nn.functional.relu(self.lay1(x))
            x2 = torch.nn.functional.relu(self.lay2(x1))
            xd = self.do(x2)
            x3 = torch.nn.functional.relu(self.lay3(xd))
            x4 = torch.nn.functional.softmax(self.lay4(x3), dim=1)
        
            # calculate regularization loss
            reg_loss = 0.0
            for param in self.parameters():
                reg_loss += 0.5 * self.reg_lambda * torch.sum(param ** 2)
        
            return x4 + reg_loss
        
# Initialize our neural network model with input and output dimensions
model = mnistClassification(input_dim=784,hdim1=1000,hdim2=666, hdim3=333,output_dim=10, reg_lambda=0.001)

# Define the learning rate and epoch 
learning_rate=1e-4
epochs = 200
batchsize = 128
step_size=50
gamma=0.1

# Define loss function and optimizer
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# Define learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Placeholders for training loss and validation accuracy during training
# Training loss should be tracked for each iteration (1 iteration -> single forward pass to the network)
# Validation accuracy should be evaluated every 'Epoch' (1 epoch -> full training dataset)
# If using batch gradient, 1 iteration = 1 epoch

train_loss_list = np.array(np.zeros((epochs,)))
validation_accuracy_list=[]
validation_accuracy_list=np.array(validation_accuracy_list)

import tqdm


# Convert the training, validation, testing dataset (NumPy arrays) into torch tensors
# Training Loop ---------------------------------------------------------------------------------------

for epoch in tqdm.trange(epochs):
    
    optimizer.zero_grad()
    train_outputs=model(XTrf)#.float())

    #train_outputs.requires_grad_(True)

    loss=loss_func(train_outputs,YTr)
    train_loss_list[epoch]=loss.item()
    
    # loss.requires_grad_(True)
    loss.backward()
    optimizer.step()
    
    scheduler.step()

    # Compute Validation Accuracy ----------------------------------------------------------------------
    
    with torch.no_grad():
        validation_outputs = model(XVF)
        correct = (torch.argmax(validation_outputs, dim=1) == torch.argmax(YV)).type(torch.FloatTensor)
        validation_accuracy_list=np.append(validation_accuracy_list,correct.mean())


# Import seaborn for prettier plots

import seaborn as sns

# Visualize training loss

plt.figure(figsize = (12, 6))

# Visualize training loss with respect to iterations (1 iteration -> single batch)
plt.subplot(2, 1, 1)
plt.plot(train_loss_list, linewidth = 3)
plt.ylabel("training loss")
plt.xlabel("epochs")
sns.despine()

# Visualize validation accuracy with respect to epochs
plt.subplot(2, 1, 2)
plt.plot(validation_accuracy_list, linewidth = 3, color = 'gold')
plt.ylabel("validation accuracy")
sns.despine()

# Compute the testing accuracy 
test_accuracy = 0.0
with torch.no_grad():
    test_outputs = model(torch.from_numpy(test1dS).float())
    test_predicted_labels = torch.argmax(test_outputs, dim=1)
    test_accuracy = (test_predicted_labels == torch.from_numpy(y_test).long()).type(torch.FloatTensor).mean()

print(f"Testing Accuracy: {test_accuracy.item()*100}%")

