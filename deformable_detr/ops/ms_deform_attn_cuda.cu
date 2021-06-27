#include "paddle/extension.h"
#include "ms_deform_im2col_cuda.cu"
#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")







std::vector<paddle::Tensor> ms_deform_attn_cuda_forward(
    const paddle::Tensor& value,
    const paddle::Tensor& spatial_shapes,
    const paddle::Tensor& level_start_index,
    const paddle::Tensor& sampling_loc,
    const paddle::Tensor& attn_weight,
    const int im2col_step
) {
    std::vector<int64_t> val_shape = value.shape();
    const int batch = val_shape[0];
    const int spatial_size = val_shape[1];
    const int num_heads = val_shape[2];
    const int channels = val_shape[3];

    std::vector<int64_t> spatial_shape = spatial_shapes.shape();
    const int num_levels = spatial_shape[0];

    std::vector<int64_t> loc_shape = sampling_loc.shape();
    const int num_query = loc_shape[1];
    const int num_point = loc_shape[4];

    //printf("==========> %d %d %d %d %d %d %d\n", batch, spatial_size, num_heads, channels, num_levels, num_query, num_point);

    const int im2col_step_ = std::min(batch, im2col_step);
    PD_CHECK(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    const int batch_n = im2col_step_;
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;

    std::vector<int64_t> out_shape{batch/im2col_step_, batch_n, num_query, num_heads, channels};
    auto out = paddle::Tensor(paddle::PlaceType::kGPU);
    out.reshape(out_shape);

    //printf("====> step1 %d %d %d %d %d\n", batch/im2col_step_, batch_n, num_query, num_heads, channels);
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        printf("-----> %d, %d \n", n , batch/im2col_step_);
        PD_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_forward_cuda", ([&] {
            ms_deformable_im2col_cuda(value.stream(),
                value.data<data_t>() + n * im2col_step_ * per_value_size,
                spatial_shapes.data<int64_t>(),
                level_start_index.data<int64_t>(),
                sampling_loc.data<data_t>() + n * im2col_step_ * per_sample_loc_size,
                attn_weight.data<data_t>() + n * im2col_step_ * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                out.mutable_data<data_t>(value.place()) + n * batch_n * num_query * num_heads * channels);
        }));
    }

    std::vector<int64_t> end_shape{batch, num_query, num_heads*channels};
    //printf("==========> %d %d %d \n", batch, num_query, num_heads*channels);

    out.reshape(end_shape);
    return {out};
}

std::vector<paddle::Tensor> ms_deform_attn_cuda_backward(
    const paddle::Tensor& value,
    const paddle::Tensor& spatial_shapes,
    const paddle::Tensor& level_start_index,
    const paddle::Tensor& sampling_loc,
    const paddle::Tensor& attn_weight,
    const paddle::Tensor& grad_output,
    const int im2col_step
) {
    //printf("start backwarding....\n");
    std::vector<int64_t> val_shape = value.shape();
    const int batch = val_shape[0];
    const int spatial_size = val_shape[1];
    const int num_heads = val_shape[2];
    const int channels = val_shape[3];

    std::vector<int64_t> spatial_shape = spatial_shapes.shape();
    const int num_levels = spatial_shape[0];

    std::vector<int64_t> loc_shape = sampling_loc.shape();
    const int num_query = loc_shape[1];
    const int num_point = loc_shape[4];

    const int im2col_step_ = std::min(batch, im2col_step);
    PD_CHECK(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto grad_value = paddle::Tensor(paddle::PlaceType::kGPU);
    grad_value.reshape(value.shape());
    auto grad_sampling_loc = paddle::Tensor(paddle::PlaceType::kGPU);
    grad_sampling_loc.reshape(sampling_loc.shape());
    auto grad_attn_weight = paddle::Tensor(paddle::PlaceType::kGPU);
    grad_attn_weight.reshape(attn_weight.shape());

    const int batch_n = im2col_step_;
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;

    //auto* grad_val_ptr =  grad_value.mutable_data<data_t>(value.place(), value.size());
    //auto* grad_sample_ptr =  grad_sampling_loc.mutable_data<data_t>(sampling_loc.place(), sampling_loc.size());
    //auto* grad_atten_ptr =  grad_attn_weight.mutable_data<data_t>(attn_weight.place(), attn_weight.size());

    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        PD_DISPATCH_FLOATING_TYPES(sampling_loc.type(), "ms_deform_attn_backward_cuda", ([&] {
            ms_deformable_col2im_cuda(value.stream(),
                                    grad_output.data<data_t>() + n * batch_n * num_query * num_heads * channels,
                                    value.data<data_t>() + n * im2col_step_ * per_value_size,
                                    spatial_shapes.data<int64_t>(),
                                    level_start_index.data<int64_t>(),
                                    sampling_loc.data<data_t>() + n * im2col_step_ * per_sample_loc_size,
                                    attn_weight.data<data_t>() + n * im2col_step_ * per_attn_weight_size,
                                    batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                                    grad_value.mutable_data<data_t>(value.place()) +  n * im2col_step_ * per_value_size,
                                    grad_sampling_loc.mutable_data<data_t>(sampling_loc.place()) + n * im2col_step_ * per_sample_loc_size,
                                    grad_attn_weight.mutable_data<data_t>(attn_weight.place()) + n * im2col_step_ * per_attn_weight_size);

        }));
    }

    return {
        grad_value, grad_sampling_loc, grad_attn_weight
    };
}