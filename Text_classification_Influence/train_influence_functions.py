import torch
from torchtext.datasets import AG_NEWS
import string
import numpy as np
from torch import nn
from transformers import BertModel
from transformers import BertTokenizer
from torch.optim import Adam
from tqdm import tqdm
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

train=AG_NEWS(root='./data', split='train')
test=AG_NEWS(root='./data', split='test')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        realtexts, labels = [], []
        for label, text in dataset:
            realtexts.append(text.translate(str.maketrans('', '', string.punctuation)))
            if label==4:
                labels.append(0)
            else:
                labels.append(label)
        self.labels = labels
        self.realtexts=realtexts
        self.texts = [tokenizer(text, padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in self.realtexts]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

trainset=Dataset(train)
testset=Dataset(test)

def load_data():

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

def train(model, trainloader, testloader, learning_rate, epochs):


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(trainloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in testloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(trainloader.dataset): .3f} \
                | Train Accuracy: {total_acc_train / len(trainloader.dataset): .3f} \
                | Val Loss: {total_loss_val / len(testloader.dataset): .3f} \
                | Val Accuracy: {total_acc_val / len(testloader.dataset): .3f}')


            
def save_model(net):
    PATH = './bert_agnews.pth'
    torch.save(net.state_dict(), PATH)
    
    
def load_model():
    PATH = './bert_agnews.pth'
    net = BertClassifier()
    net.load_state_dict(torch.load(PATH))
    return net



if __name__ == "__main__":
    EPOCHS = 10
    LR = 1e-6
    trainloader, testloader = load_data()
    model = BertClassifier()
    train(model, trainloader, testloader, LR, EPOCHS)
    save_model(model)