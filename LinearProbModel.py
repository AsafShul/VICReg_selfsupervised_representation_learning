import torch
from torch import nn
from tqdm import tqdm
from BaseModel import BaseModel


class LinearProbModel(BaseModel):
    # training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    TRAIN_EPOCHS_NUM = 10
    BETAS = (0.9, 0.999)
    WEIGHT_DECAY = 1e-6
    LOG_INTERVAL = 10

    MODEL_NAME = 'LinearProbModel.pth'
    TRAINED_MODELS_DIR = 'trained_models'

    def __init__(self, input_dim, output_dim, batch_size=BATCH_SIZE,
                 learning_rate=LEARNING_RATE, train_epochs_num=TRAIN_EPOCHS_NUM,
                 betas=BETAS, weight_decay=WEIGHT_DECAY):
        super(LinearProbModel, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_epochs_num = train_epochs_num
        self.classes = None

        self.fc = nn.Linear(input_dim, output_dim)
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        return self.fc(batch[0].to(self.device))

    def predict(self, x):
        assert self.classes is not None, 'Model is not trained yet'
        return self.classes[torch.argmax(self.forward(x), dim=1)]

    def fit(self, train_loader, save_model=True):
        self.train()
        print('setting classes.')
        self.classes = train_loader.dataset.dataset.classes
        print(f'Training {self.MODEL_NAME.split(".")[0]}...')
        for epoch in range(self.train_epochs_num):
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self.forward(batch)
                loss = self.criterion(output, batch[train_loader.dataset.CLASSES_IDX].to(self.device))
                loss.backward()
                self.optimizer.step()

                if batch_idx % self.LOG_INTERVAL == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(batch[0])}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        print('Finished Training')
        if save_model:
            self.save_model()

    def evaluate(self, test_loader):
        correct = 0
        print('Evaluating...')
        self.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                output = self.forward(batch)
                pred = torch.argmax(output, dim=1)
                correct += pred.eq(batch[test_loader.dataset.CLASSES_IDX].view_as(pred)).sum().item()
        acc = correct / len(test_loader.dataset)
        print(f'  - acc of linear probing on test dataset: {acc:.4f}')
        return acc
