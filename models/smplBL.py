'''
This file includes the light weighted fc models on pose data directly including
Simple_baseline
SPA for AHuP.
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn

## add the statics for SURREAL and ScanAva
# the first pelvis 3rd entry is 0 originally, 1 to avoid dividing by 0
dct_mu= {
    'ScanAva': [[32.0384521484375, 33.55183410644531, 32.0], [32.12306213378906, 33.46104049682617, 31.991485595703125], [32.17890167236328, 44.99711990356445, 31.274211883544922], [32.14009094238281, 58.50051498413086, 30.330101013183594], [31.954179763793945, 33.642765045166016, 32.00851058959961], [31.982545852661133, 45.324398040771484, 31.29669761657715], [31.966041564941406, 58.71873474121094, 30.354013442993164], [32.05836486816406, 24.03989601135254, 32.77383041381836], [32.078304290771484, 14.527924537658691, 33.54777145385742], [32.08232116699219, 11.724708557128906, 33.77440643310547], [32.09056854248047, 6.1173248291015625, 34.227874755859375], [31.952795028686523, 16.409561157226562, 33.40259552001953], [31.923128128051758, 21.881343841552734, 33.032798767089844], [31.966999053955078, 22.85951042175293, 33.06393814086914], [32.192047119140625, 16.132776260375977, 33.36924362182617], [32.28194046020508, 21.44119644165039, 32.9817008972168], [32.31734848022461, 21.704811096191406, 33.052310943603516]],
    'SURREAL':[[31.932945251464844, 27.786067962646484, 32.0], [31.98326873779297, 30.90540885925293, 32.019554138183594], [32.001441955566406, 42.66286849975586, 32.02734375], [31.994237899780273, 55.46914291381836, 32.041168212890625], [31.880756378173828, 30.82353973388672, 31.988588333129883], [31.817941665649414, 42.687477111816406, 31.986968994140625], [31.858552932739258, 55.32418441772461, 31.98884391784668], [31.951688766479492, 18.467164993286133, 31.981653213500977], [31.966354370117188, 11.429707527160645, 31.959640502929688], [31.960851669311523, 9.897293090820312, 31.958826065063477], [31.949817657470703, 6.83167839050293, 31.957931518554688], [31.83858299255371, 14.42465877532959, 31.930740356445312], [31.76633644104004, 21.028287887573242, 31.937170028686523], [31.724340438842773, 25.191682815551758, 31.95848846435547], [32.08332443237305, 14.444685935974121, 32.002193450927734], [32.12047576904297, 21.434995651245117, 32.03466033935547], [32.093780517578125, 25.704139709472656, 32.053916931152344]]
}
dct_std= {
    'ScanAva': [[6.266915798187256, 5.595676422119141, 1.0], [7.298173427581787, 5.6444926261901855, 2.6398062705993652], [6.765401363372803, 5.427472114562988, 5.656977653503418], [9.218626022338867, 6.260773658752441, 6.276914119720459], [6.790135860443115, 5.595905303955078, 2.639806032180786], [6.991155624389648, 5.383211612701416, 6.270663261413574], [9.446077346801758, 6.3341593742370605, 6.8559699058532715], [4.026186943054199, 4.053257942199707, 3.1699256896972656], [5.132321357727051, 2.9246509075164795, 6.339852809906006], [5.803759574890137, 2.501185417175293, 7.189228534698486], [7.58555793762207, 2.0597336292266846, 8.95622444152832], [6.381109714508057, 3.42977237701416, 6.823390483856201], [7.622098445892334, 4.828530311584473, 8.779298782348633], [9.270851135253906, 10.563284873962402, 11.215736389160156], [6.713460445404053, 3.5882468223571777, 6.471081256866455], [7.83514404296875, 5.0148515701293945, 8.297295570373535], [9.553818702697754, 11.141189575195312, 10.901069641113281]],
    'SURREAL':[[5.666924476623535, 3.596813440322876, 1.0], [6.19535493850708, 3.665545701980591, 1.6486741304397583], [7.129264831542969, 3.7862250804901123, 4.832751750946045], [8.686056137084961, 5.460014820098877, 5.919955253601074], [6.136964797973633, 3.610255479812622, 1.6448414325714111], [7.178542613983154, 3.9472546577453613, 4.620211124420166], [8.786582946777344, 5.598336219787598, 5.826437473297119], [5.199839115142822, 3.450141668319702, 2.1289451122283936], [5.969222545623779, 4.295408248901367, 3.792048692703247], [6.244235992431641, 4.519399166107178, 4.592365741729736], [7.94729471206665, 5.8209099769592285, 6.833676338195801], [7.175869464874268, 4.079746723175049, 5.158937454223633], [9.032966613769531, 5.2870001792907715, 7.129406929016113], [10.323783874511719, 8.898763656616211, 8.943641662597656], [7.291187763214111, 4.02107572555542, 5.09914493560791], [9.146483421325684, 5.126472473144531, 7.156311511993408], [10.209549903869629, 8.408149719238281, 9.176583290100098]]
}


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class PA_D(nn.Module):
    def __init__(self, n_jt=17, d=3):
        super(PA_D, self).__init__()

        self.n_jt = n_jt
        self.D = nn.Sequential()
        self.D.add_module('d_fc1', nn.Linear(n_jt*d, 100))   # 51  mismatch m1 10 x 150 , 10 x 1
        self.D.add_module('d_bn1', nn.BatchNorm1d(100))
        self.D.add_module('d_relu1', nn.ReLU(True))
        self.D.add_module('d_fc2', nn.Linear(100, 50))
        self.D.add_module('d_bn2', nn.BatchNorm1d(50))
        self.D.add_module('d_relu2', nn.ReLU(True))
        self.D.add_module('d_fc3', nn.Linear(50, 1))   # BCE will handle, signle output +inf -inf score sigmoid to 0 and 1 BCEwithLogits
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax())

    def forward(self, x):
        return self.D(x)

# make another one for SBL for release later
class SimpSBL(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5,
                 d=2,
                 n_jt=17, mode=1):  # except pelvis
        super(SimpSBL, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.mode = mode  #

        # 2d joints
        self.input_size = n_jt * d  # default 16, here we use 17 for full h36m
        # 3d joints
        self.output_size = n_jt * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        # print('x size', x.size())
        y = self.w1(x)
        # print('y size', y.size())
        # print('bch1', self.batch_norm1)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)
        if self.mode == 1:
            return y
        elif self.mode == 2:
            return y + x
        else:
            raise ValueError('{} not in available nodes [1,2]'.format(self.mode))

#  later change the paNet  ( pose
class PAnet(nn.Module):
    ''' The SPA net'''
    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5,
                 d=2,
                 n_jt=17, mode=1):  # except pelvis, 1 for direct map
        super(PAnet, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.mode = mode        #

        # 2d joints
        self.input_size =  n_jt * d     # default 16, here we use 17 for full h36m
        # 3d joints
        self.output_size = n_jt * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        # print('x size', x.size())
        y = self.w1(x)
        # print('y size', y.size())
        # print('bch1', self.batch_norm1)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)
        if self.mode == 1:
            return y
        elif self.mode == 2:
            return y+x
        else:
            raise ValueError('{} not in available nodes [1,2]'.format(self.mode))
