#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Leyang 
# time:2020/4/10

with open("decode_sample.txt", "r") as f:
    files = f.readlines()

assert len(files) == 886
r1, r2, mrr = 0, 0, 0
for line in files:
    golden = "C" # fake answer
    output_list = line.strip().split("\t")[1:]
    assert sorted(output_list) == ['A', 'B', 'C', 'D']
    index = output_list.index(golden)
    if index == 0:
        r1 += 1
    elif index == 1:
        r2 += 1
    mrr += 1 / (index + 1)
print("R@1: %.3f \t R@2: %.3f \t MRR %.3f" %(r1/886, (r1/886 + r2/886), mrr/886))