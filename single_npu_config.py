# 单NPU训练配置
run = "single_npu_fineweb_test"
depth = 6
device_batch_size = 4
total_batch_size = 8192
num_iterations = 100
embedding_lr = 0.001
unembedding_lr = 0.0001
matrix_lr = 0.0005
grad_clip = 1.0
eval_every = 50
sample_every = 999999
core_metric_every = 999999