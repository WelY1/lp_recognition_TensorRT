import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters() 
        '''
        # 将 LSTM 层中的权重和偏置参数 "扁平化"，将其从多维数组转换为一维数组，并将其保存在 LSTM 层对象中，
        以便在后续的前向传播中可以重复使用，从而避免了内存重新分配，提高了计算性能。
        这个函数需要在使用 LSTM 层前调用，以确保 LSTM 层的参数在前向传播时被正确地扁平化，从而获得最佳的性能。
        '''
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
