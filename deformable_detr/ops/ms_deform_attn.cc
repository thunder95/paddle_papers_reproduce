#include "paddle/extension.h"

#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")


// 维度推导
std::vector<std::vector<int64_t>> ReluInferShape(
    std::vector<int64_t> value_shape,
    std::vector<int64_t> spatial_shapes_shape,
    std::vector<int64_t> level_start_index_shape,
    std::vector<int64_t> sampling_loc_shape,
    std::vector<int64_t> grad_output_shape
) {
    const int batch = value_shape[0];
    const int num_heads = value_shape[2];
    const int channels = value_shape[3];
    const int num_query = sampling_loc_shape[1];
    std::vector<int64_t> out_shape{batch, num_query, num_heads*channels};
    return {out_shape};
}

// 类型推导
std::vector<paddle::DataType> ReluInferDtype(
    paddle::DataType value_dtype,
    paddle::DataType spatial_shapes_dtype,
    paddle::DataType level_start_index_dtype,
    paddle::DataType sampling_loc_dtype,
    paddle::DataType attn_weight_dtype
) {
    return {value_dtype};
}
std::vector<paddle::Tensor> ms_deform_attn_cuda_forward(
    const paddle::Tensor& value,
    const paddle::Tensor& spatial_shapes,
    const paddle::Tensor& level_start_index,
    const paddle::Tensor& sampling_loc,
    const paddle::Tensor& attn_weight,
    const int im2col_step);


std::vector<paddle::Tensor> ms_deform_attn_cuda_backward(
    const paddle::Tensor& value,
    const paddle::Tensor& spatial_shapes,
    const paddle::Tensor& level_start_index,
    const paddle::Tensor& sampling_loc,
    const paddle::Tensor& attn_weight,
    const paddle::Tensor& grad_output,
    const int im2col_step);

std::vector<paddle::Tensor> MsDeformAttnCUDAForward(
    const paddle::Tensor& value,
    const paddle::Tensor& spatial_shapes,
    const paddle::Tensor& level_start_index,
    const paddle::Tensor& sampling_loc,
    const paddle::Tensor& attn_weight,
    const int im2col_step) {

    CHECK_INPUT(value);
    CHECK_INPUT(spatial_shapes);
    CHECK_INPUT(level_start_index);
    CHECK_INPUT(sampling_loc);
    CHECK_INPUT(attn_weight);

    return ms_deform_attn_cuda_forward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
}

std::vector<paddle::Tensor> MsDeformAttnCUDABackward(
    const paddle::Tensor& value,
    const paddle::Tensor& spatial_shapes,
    const paddle::Tensor& level_start_index,
    const paddle::Tensor& sampling_loc,
    const paddle::Tensor& attn_weight,
    const paddle::Tensor& grad_output,
    const int im2col_step
) {
    CHECK_INPUT(value);
    CHECK_INPUT(spatial_shapes);
    CHECK_INPUT(level_start_index);
    CHECK_INPUT(sampling_loc);
    CHECK_INPUT(attn_weight);
    CHECK_INPUT(grad_output);
    return ms_deform_attn_cuda_backward(value, spatial_shapes, level_start_index,  sampling_loc, attn_weight, grad_output, im2col_step);
}

PD_BUILD_OP(custom_ms_deform_attn)
    .Inputs({"value", "spatial_shapes", "level_start_index", "sampling_loc", "attn_weight"})
    .Outputs({"out"})
    .Attrs({"im2col_step: int"})
    .SetKernelFn(PD_KERNEL(MsDeformAttnCUDAForward))
    .SetInferShapeFn(PD_INFER_SHAPE(ReluInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ReluInferDtype));

PD_BUILD_GRAD_OP(custom_ms_deform_attn)
    .Inputs({"value", "spatial_shapes", "level_start_index", "sampling_loc", "attn_weight", paddle::Grad("out")})
    .Outputs({paddle::Grad("value"), paddle::Grad("sampling_loc"), paddle::Grad("attn_weight")})
    .Attrs({"im2col_step: int"})
    .SetKernelFn(PD_KERNEL(MsDeformAttnCUDABackward));