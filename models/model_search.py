import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
from models.operations import *
from torch.autograd import Variable
from extras.genotypes import PRIMITIVES
from extras.genotypes import Genotype


class MixedOp(nn.Module):

    def __init__(self, C, stride, switch, edge):
        super(MixedOp, self).__init__()
        self.m_ops = nn.ModuleList()
        #self.m_ops = nn.Sequential()
        for i in range(len(switch)):
            if switch[i] == 100:
                primitive = 'none'
            else:
                primitive = PRIMITIVES[switch[i]]
            op = OPS[primitive](C, stride, False, edge)
            #seq = nn.Sequential()
            #seq.add_module(primitive+'_'+str(i), op)
            if 'pool' in primitive:
                op = nn.Sequential(OrderedDict([('mixedop_pool_edge-'+str(edge), op), ('mixedop_pool_bn_edge-'+str(edge), nn.BatchNorm2d(C, affine=False))]))
            else:
                op = nn.Sequential(OrderedDict([('mixedop_edge-'+str(edge), op)]))
            self.m_ops.add_module('oper-'+str(switch[i]), op)
            
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.m_ops))

class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, switches):
        super(Cell, self).__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, edge='preproc0-red', affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, edge='preproc0-nor', affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, edge='preproc1', affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        switch_count = 0
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, switch=switches[switch_count], edge=switch_count)
                self.cell_ops.append(op)
                #self.cell_ops.add_module('edge', op)
                switch_count = switch_count + 1
    
    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self.cell_ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._steps:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, switches_normal=[]):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.switches_normal = switches_normal

        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(OrderedDict([('stem_conv',
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False)),('stem_bn',
            nn.BatchNorm2d(C_curr))])
        )
    
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_normal)
            else:
                reduction = False
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_normal)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._reset_alphas(self.switches_normal)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            weights = self.alphas_normal
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

    def _reset_alphas(self, switches):
        k = len(switches)
        self.alphas_normal = []
        for i in range(k):
            self.alphas_normal.append([1. for j in range(len(switches[i]))])

        self._arch_parameters = [self.alphas_normal]

    def _init_switches(self):
        switches = []
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        for i in range(k):
            switches.append([j for j in range(len(PRIMITIVES))])
        return switches

    def _perturb_binary(self, edge, op, remove=False):
        if remove:
            self.alphas_normal[edge][op] = 0.
        else:
            self.alphas_normal[edge][op] = 1.

    def arch_parameters(self):
        return self._arch_parameters
