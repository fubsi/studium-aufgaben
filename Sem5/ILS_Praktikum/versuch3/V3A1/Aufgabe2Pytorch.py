import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, D, M, K):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(D, M)  # Input layer to hidden layer
        self.fc2 = nn.Linear(M, K)   # Hidden layer to output layer

        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Move model to the appropriate device

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            output = self.forward(x)
            return output
    
# Function to train the neural network
def train_neural_network(D, M, K, X_train, T_train, X_validate, T_validate, num_epochs=1000, learning_rate=0.01):
    print(f"Training with D={D}, M={M}, K={K}, num_epochs={num_epochs}, learning_rate={learning_rate}")
    # Create the model
    model = SimpleNN(D,M,K)

    # Convert inputs and targets to tensors
    inputs = torch.tensor(X_train, dtype=torch.float32)
    targets = torch.tensor(T_train, dtype=torch.float32)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop with cross-validation
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        # Shuffle the inputs and targets for each epoch
        indices = np.random.permutation(len(inputs))
        inputs = inputs[indices]
        targets = targets[indices]
        # Iterate over each sample in the batch
        
        for i in range(len(inputs)):
            # Get the input and target for the current sample
            input_sample = inputs[i].unsqueeze(0)
            target_sample = targets[i].unsqueeze(0)

            # To device
            input_sample = input_sample.to(model.device)
            target_sample = target_sample.to(model.device)


            # Forward pass
            outputs = model(input_sample)

            # Compute the loss
            loss = criterion(outputs, target_sample)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')
        if (epoch + 1) % 200 == 0:
            # Validate
            model.eval()
            with torch.no_grad():
                val_losses = []
                for i in range(15):  # Print 5 random validation samples
                    random_idx = np.random.choice(len(X_validate), 1)
                    val_input = torch.tensor(X_validate[random_idx], dtype=torch.float32)
                    val_target = torch.tensor(T_validate[random_idx], dtype=torch.float32)
                    val_input = val_input.to(model.device)
                    val_target = val_target.to(model.device)
                    val_outputs = model.predict(val_input).cpu().numpy()  # Move output back to CPU for printing
                    val_loss = criterion(torch.tensor(val_outputs, dtype=torch.float32), val_target)
                    val_losses.append(val_loss.item())
            print(f'Last Choice: {np.argmax(val_outputs)}, Last Target: {np.argmax(val_target)}, Validation Loss: {np.mean(val_losses):.4f}')
            print(f'Output: {val_outputs}, Target: {val_target.numpy()}')
    
    return model

def create_data(path=None):
    # (i) Create training data
    forestdata = pd.read_csv(path); # load data as pandas data frame 
    classlabels = ['s','h','d','o'];                                      # possible class labels (C=4) 
    classidx = {classlabels[i]:i for i in range(len(classlabels))}     # dict for mapping classlabel to index 
    C = len(classlabels)        # number of classes (Note: K is now the number of nearest-neighbors!!!!!!)
    T_txt = forestdata.values[:,0]           # array of class labels of data vectors (class label is first data attribute)
    X = forestdata.values[:,1:]           # array of feature vectors (features are remaining attributes)
    T = [classidx[t.strip()] for t in T_txt]          # transform text labels 's','h','d','o' to numeric lables 0,1,2,3
    X,T=np.array(X,'float'),np.array(T,'int')  # convert to numpy arrays

    # X1 = np.array([[-2,-1], [-2,2], [-1.5,1], [0,2], [2,1], [3,0], [4,-1], [4,2]])  # class 1 data
    # N1,D1 = X1.shape
    # T1 = np.array(N1*[[1.,0]])     # corresponding class labels with one-hot coding: [1,0]=class 1;
    # X2 = np.array([[-1,-2],[-0.5,-1],[0,0.5],[0.5,-2],[1,0.5],[2,-1],[3,-2]])       # class 2 data
    # N2,D2 = X2.shape
    # T2 = np.array(N2*[[0,1.]])     # corresponding class labels with one-hot coding: [0,1]=class 2 
    # X = np.concatenate((X1,X2))    # entire data set
    # T = np.concatenate((T1,T2))    # entire label set
    N,D = X.shape
    newT = []
    for idx,t in enumerate(T):  # convert target vector T to one-hot coding
        if t==0: newT.append(np.array([[1,0,0,0]]))  # class 1
        elif t==1: newT.append(np.array([[0,1,0,0]]))  # class 2
        elif t==2: newT.append(np.array([[0,0,1,0]]))  # class 3
        elif t==3: newT.append(np.array([[0,0,0,1]]))  # class 4
    T = np.concatenate(newT)  # convert to numpy array
    X=np.concatenate((np.ones((N,1)),X),1)  # X is extended by a column vector with ones (for bias weights w_j0)
    N,D = X.shape                      # update size parameters
    N,K = T.shape                      # update size parameters
    print("X=",X)
    print("T=",T)
    print(X.shape, T.shape)  # print size of data and target vectors
    return X, T

if __name__ == "__main__":
    X_train,T_train = create_data('C:\\Users\\fbrze\\Documents\\EigeneDateien\\Programmierung_etc\\Git\\studium-aufgaben\\Sem5\\ILS_Praktikum\\versuch3\\V3A1\\training.csv')  # Create the data
    X_validate,T_validate = create_data('C:\\Users\\fbrze\\Documents\\EigeneDateien\\Programmierung_etc\\Git\\studium-aufgaben\\Sem5\\ILS_Praktikum\\versuch3\\V3A1\\testing.csv')  # Create the validation data
    N,D = X_train.shape  # Number of samples and features
    N,K = T_train.shape  # Number of samples and classes
    print(f"Number of samples: {N}, Number of features: {D}, Number of classes: {K}")
    M = range(5,21)  # Number of hidden neurons
    num_epochs = 1000  # Number of training epochs
    learning_rate = [1e-2,1e-3,1e-4]  # Learning rate for the optimizer

    for m in M:
        for lr in learning_rate:
            # Train the neural network
            model = train_neural_network(D, m, K, X_train, T_train, X_validate, T_validate, num_epochs, lr)