"""
This code is modified version of LCNN baseline
from ASVSpoof2021 challenge - https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-LFCC-LCNN/project/baseline_LA/model.py
"""
import sys

import torch
import torch.nn as torch_nn

from src import frontends


NUM_COEFFICIENTS = 384


# For blstm
class BLSTMLayer(torch_nn.Module):
    """ Wrapper over dilated conv1D
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    We want to keep the length the same
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        if output_dim % 2 != 0:
            print("Output_dim of BLSTMLayer is {:d}".format(output_dim))
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)
        # bi-directional LSTM
        self.l_blstm = torch_nn.LSTM(
            input_dim,
            output_dim // 2,
            bidirectional=True
        )
    def forward(self, x):
        # permute to (length, batchsize=1, dim)
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        # permute it backt to (batchsize=1, length, dim)
        return blstm_data.permute(1, 0, 2)


class MaxFeatureMap2D(torch_nn.Module):
    """ Max feature map (along 2D) 
    
    MaxFeatureMap2D(max_dim=1)
    
    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)

    
    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)
    
    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)
    
    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """
    def __init__(self, max_dim = 1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)

        shape = list(inputs.size())

        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim]//2
        shape.insert(self.max_dim, 2)

        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m


##############
## FOR MODEL
##############

class LCNN(torch_nn.Module):
    """ Model definition
    """
    def __init__(self, **kwargs):
        super().__init__()
        input_channels = kwargs.get("input_channels", 1)
        num_coefficients = kwargs.get("num_coefficients", NUM_COEFFICIENTS)

        # Working sampling rate
        self.num_coefficients = num_coefficients

        # dimension of embedding vectors
        # here, the embedding is just the activation before sigmoid()
        self.v_emd_dim = 1

        # it can handle models with multiple front-end configuration
        # by default, only a single front-end

        self.m_transform = torch_nn.Sequential(
            torch_nn.Conv2d(input_channels, 64, (5, 5), 1, padding=(2, 2)),
            MaxFeatureMap2D(),
            torch.nn.MaxPool2d((2, 2), (2, 2)),

            torch_nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            torch_nn.BatchNorm2d(32, affine=False),
            torch_nn.Conv2d(32, 96, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),

            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch_nn.BatchNorm2d(48, affine=False),

            torch_nn.Conv2d(48, 96, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            torch_nn.BatchNorm2d(48, affine=False),
            torch_nn.Conv2d(48, 128, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),

            torch.nn.MaxPool2d((2, 2), (2, 2)),

            torch_nn.Conv2d(64, 128, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            torch_nn.BatchNorm2d(64, affine=False),
            torch_nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            torch_nn.BatchNorm2d(32, affine=False),

            torch_nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            torch_nn.BatchNorm2d(32, affine=False),
            torch_nn.Conv2d(32, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            torch_nn.MaxPool2d((2, 2), (2, 2)),

            torch_nn.Dropout(0.7)
        )

        self.m_before_pooling = torch_nn.Sequential(
            BLSTMLayer((self.num_coefficients//16) * 32, (self.num_coefficients//16) * 32),
            BLSTMLayer((self.num_coefficients//16) * 32, (self.num_coefficients//16) * 32)
        )

        self.m_output_act = torch_nn.Linear((self.num_coefficients // 16) * 32, self.v_emd_dim)

    def _compute_embedding(self, x):
        """ definition of forward method 
        Assume x (batchsize, length, dim)
        Output x (batchsize * number_filter, output_dim)
        """
        # resample if necessary
        # x = self.m_resampler(x.squeeze(-1)).unsqueeze(-1)
        
        # number of sub models
        batch_size = x.shape[0]

        # buffer to store output scores from sub-models
        output_emb = torch.zeros(
            [batch_size, self.v_emd_dim],
            device=x.device,
            dtype=x.dtype
        )

        # compute scores for each sub-models
        idx = 0

        # compute scores
        #  1. unsqueeze to (batch, 1, frame_length, fft_bin)
        #  2. compute hidden features
        x = x.permute(0,1,3,2)
        hidden_features = self.m_transform(x)

        #  3. (batch, channel, frame//N, feat_dim//N) ->
        #     (batch, frame//N, channel * feat_dim//N)
        #     where N is caused by conv with stride
        hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
        frame_num = hidden_features.shape[1]

        hidden_features = hidden_features.view(batch_size, frame_num, -1)
        #  4. pooling
        #  4. pass through LSTM then summingc
        hidden_features_lstm = self.m_before_pooling(hidden_features)

        #  5. pass through the output layer
        tmp_emb = self.m_output_act((hidden_features_lstm + hidden_features).mean(1))
        output_emb[idx * batch_size : (idx+1) * batch_size] = tmp_emb

        return output_emb

    def _compute_score(self, feature_vec):
        # feature_vec is [batch * submodel, 1]
        return torch.sigmoid(feature_vec).squeeze(1)

    def forward(self, x):
        feature_vec = self._compute_embedding(x)
        return feature_vec



class FrontendLCNN(LCNN):
    """ Model definition
    """
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)

        self.device = device

        frontend_name = kwargs.get("frontend_algorithm", [])
        self.frontend = frontends.get_frontend(frontend_name)
        print(f"Using {frontend_name} frontend")

    def _compute_frontend(self, x):
        frontend = self.frontend(x)
        if frontend.ndim < 4:
            return frontend.unsqueeze(1)  # (bs, 1, n_lfcc, frames)
        return frontend # (bs, n, n_lfcc, frames)

    def forward(self, x):
        x = self._compute_frontend(x)
        feature_vec = self._compute_embedding(x)

        return feature_vec


if __name__ == "__main__":

    device = "cuda"
    print("Definition of model")
    model = FrontendLCNN(input_channels=2, num_coefficients=80, device=device, frontend_algorithm=["mel_spec"])
    model = model.to(device)
    batch_size = 12
    mock_input = torch.rand((batch_size, 64_600,), device=device)
    output = model(mock_input)
    print(output.shape)
