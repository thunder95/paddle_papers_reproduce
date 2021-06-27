
#y
# import paddle
# from paddle.utils.cpp_extension import load
#
# custom_ops = load(
#     name="custom_jit_ops",
#     sources=["ms_deform_attn.cc", "ms_deform_attn_cuda.cu", "ms_deform_im2col_cuda.cuh"])

#python ms_deform_attn_setup.py install

from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='ms_deform_attn_op',
    ext_modules=CUDAExtension(
        sources=["ms_deform_attn.cc", "ms_deform_attn_cuda.cu", "ms_deform_im2col_cuda.cu"],
        extra_compile_args={
            'nvcc': ["-arch=sm_75"]
        }
    )
)