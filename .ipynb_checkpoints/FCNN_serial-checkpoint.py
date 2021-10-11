import torch
import torch.nn as nn

class HARmodel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, num_classes,in_channels,num_features,kernel1_size,kernel2_size):
        super().__init__()
        # Extract features, 1D conv layers
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=kernel1_size,stride=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel1_size,stride=1),
            nn.ReLU(),
            #nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2,stride=1),
            
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel1_size,stride=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel1_size,stride=1),
            nn.ReLU(),
            #nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2,stride=1),
            )
        self.embedding=nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=num_features, kernel_size=kernel2_size,stride=1),
        )
        self.classifier=nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=num_classes, kernel_size=kernel2_size,stride=1),
        )        

    def forward(self, x):
        x = self.features(x)
        #embedding
        embedding=self.embedding(x)
        
        #classification
        classification=self.classifier(embedding)
        classification=torch.mean(classification,axis=2)
        
        embedding=torch.mean(embedding,axis=2)
        
        return embedding,classification