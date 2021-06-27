import paddle
# from ms_deform_attn_op import custom_ms_deform_attn
import numpy as np
import pickle
from paddle.nn.functional import mse_loss

im2col_step = 2

# 加载上次保存
with open('/f/hulei/pd_match/detr/Deformable-DETR/models/ops/save_new.pkl', 'rb') as f:
    saved = pickle.load(f)

value = paddle.to_tensor(saved["value"], stop_gradient=False)
sampling_locations = paddle.to_tensor(saved["sampling_locations"], stop_gradient=False)
attention_weights = paddle.to_tensor(saved["attention_weights"], stop_gradient=False)
level_start_index = paddle.to_tensor(saved["level_start_index"])
shapes = paddle.to_tensor(saved["shapes"])

# output_cuda = custom_ms_deform_attn(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step)
# print(output_cuda.numpy, saved["output_cuda"])
# np.testing.assert_allclose(output_cuda, saved["output_cuda"], rtol=1e-5, atol=0)


from paddle.utils.cpp_extension import load
custom_ops = load(
    name="custom_jit_ops",
    sources=["ms_deform_attn.cc", "ms_deform_attn_cuda.cu", "ms_deform_im2col_cuda.cu"],
    extra_cuda_cflags=["-arch=sm_75"])

print("value type", type(value))
print("shapes type", type(shapes))
print("level_start_index type", type(level_start_index))
print("sampling_locations type", type(sampling_locations))
print("attention_weights type", type(attention_weights))
print("im2col_step type", type(im2col_step))

output_cuda = custom_ops.custom_ms_deform_attn(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step)
print("paddle output", output_cuda.numpy())
print("torch output: ", saved["output_cuda"])

np.testing.assert_allclose(output_cuda.numpy(), saved["output_cuda"], rtol=1e-5, atol=0)
dummy_output = paddle.ones_like(output_cuda)
mse = mse_loss(output_cuda, dummy_output)
mse.backward()
print(np.testing.assert_allclose(value.grad, saved["value_grad"], rtol=1e-5, atol=0)) #有较明显的差异：0.0005

