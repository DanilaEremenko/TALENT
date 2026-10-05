import math
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_out: int, 
        d_layers: ty.List[int],    
        dropout: float,
        
        ) -> None:
        super().__init__()
        self.dropout = dropout
        self.d_out = d_out 
        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)


    def forward(self, x, x_cat = None, return_embs=False):
        # x_cat is already numerically encoded (e.g. tabr_ohe) when it is not None,
        # so it is simply concatenated with the numerical features.
        if x is None:
            x = x_cat.to(torch.get_default_dtype()) if x_cat.dtype in (torch.int32, torch.int64) else x_cat
        elif x_cat is not None:
            x = torch.cat([x, x_cat.to(x.dtype)], dim=1)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        logit = self.head(x)        
        if self.d_out == 1:
            logit = logit.squeeze(-1)

        if return_embs:
            return logit, x
        else:
            return  logit