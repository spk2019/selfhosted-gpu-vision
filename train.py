import torch 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
#
print("cuda version",torch.version.cuda)
batch_size=64

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)



import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    test loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_test_loss=float('inf')):
        self.best_test_loss = best_test_loss
        
    def __call__(self, current_test_loss, epoch, model, optimizer, criterion):

        if current_test_loss < self.best_test_loss:
            self.best_test_loss = current_test_loss
            print(f"Best test loss: {self.best_test_loss}")
            print(f"Saving the model ........")
            
            torch.save(model,"artifacts/model.pkl")





device = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    

model = Net().to(device)
loss_function = nn.CrossEntropyLoss()
#optimizer
optimizer = optim.Adam(model.parameters(),lr=0.01)
# initialize SaveBestModel class
save_best_model = SaveBestModel()



def train(dataloader,model,loss_function,optimizer):
    size = len(dataloader.dataset)
    acc = 0
    for batch , (x,y) in enumerate(dataloader):
        x,y = x.to(device), y.to(device)
        outputs = model(x)
        loss = loss_function(outputs,y)
        _, predicted = torch.max(outputs.data, 1)
        acc += (predicted == y).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if batch % 100 == 0 :
            print("Train loss for this epoch : {:.4f}   [{}/{}]".format(loss.item(),batch * len(x),size))  # print when batch = 100 , each batch contains 64 points , so total 6400
    train_accuracy = acc / size
    return loss.item(),train_accuracy      


def test(dataloader,model):
    size = len(dataloader.dataset)
    model.eval()
    acc = 0
    with torch.no_grad():
        
        for batch , (x,y) in enumerate(dataloader):
            x,y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_function(outputs,y)
            _, predicted = torch.max(outputs.data, 1)
        
            acc += (predicted == y).sum().item()
            
        test_accuracy =  acc / size  
        print(f"Test accuracy for this epoch : {test_accuracy:.2f}")
    return loss.item(),test_accuracy


epochs =2
test_accuracy = []
train_accuracy= []
test_loss=[]
train_loss=[]
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}\n-------------------------------")
    train_loss_epoch,train_accuracy_epoch = train(train_loader, model, loss_function, optimizer)
    test_loss_epoch,test_accuracy_epoch = test(test_loader, model)
    
    train_loss.append(train_loss_epoch)
    test_loss.append(test_loss_epoch)
    train_accuracy.append(train_accuracy_epoch)
    test_accuracy.append(test_accuracy_epoch)

    
    save_best_model(test_loss_epoch, epoch, model, optimizer, loss_function)
    
#plot accuracy
plt.plot(train_accuracy,color='green', linestyle='-', label='train accuracy')
plt.plot(test_accuracy, color='blue', linestyle='-', label='test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('artifacts/accuracy.png')
    


