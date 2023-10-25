import torch 
import torch.nn as nn

# torch.save(model, PATH)
# model = torch.load(PATH)
# model.eval()


class Model (nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)


    def forward(self, x):
        out = self.linear(x)
        y_pred = torch.sigmoid(out)
        return y_pred

model = Model(n_input_features=6)

for param in model.parameters():
    print(param)

FILE = "model.pth"
torch.save(model.state_dict(), FILE)

#model = torch.load(FILE)
#model.eval()

loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

for param in loaded_model.parameters():
    print(param)