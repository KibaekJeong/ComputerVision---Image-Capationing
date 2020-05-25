import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        #Embedding layer which takes in words and create vector of embedding size
        self.embedding = nn.Embedding(vocab_size,embed_size)
        
        #LSTM, embedded vector as input and output hidden states 
        #initialized with dropout rate of 0.5
        self.lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size, num_layers = num_layers,batch_first=True,dropout= 0.5)
        #fully connected layer 
        self.fc = nn.Linear(hidden_size,vocab_size)
        #initialize weights
        self.init_weights()
    
    
    def forward(self, features, captions):
        #remove 'end' token from caption
        captions = captions[:,:-1]
        #get embedding vector corresponding to the caption
        captions = self.embedding(captions)
        
        #unsqueeze feature so that shape matches with captions
        features = features.unsqueeze(1)
        
        #concatenate feature and caption, which is the input for lstm
        inputs = torch.cat((features,captions),dim = 1)
        
        #lstm prediction
        out,hidden_state = self.lstm(inputs)
        
        #LSTM prediction to word prediction
        out = self.fc(out)
        return out
        
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1,0.1)
        self.fc.weight.data.uniform_(-0.1,0.1)
        self.fc.bias.data.fill_(0)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        
        prediction = []
        
        while True:
            lstm_out,states = self.lstm(inputs,states)
            
            output = self.fc(lstm_out)
            
            output = output.squeeze(1)
            
            _,max_ind = torch.max(output,dim=1)
            
            prediction.append(max_ind.item())
            if (max_ind == 1 or len(output)>= max_len):
                break
            inputs = self.embedding(max_ind)
            inputs = inputs.unsqueeze(1)
        
         
        return prediction
    
            