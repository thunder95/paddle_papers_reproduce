from backbone import build_backbone
from transformer import build_transformer
import paddle.nn as nn
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from scipy.optimize import linear_sum_assignment
from util.box_ops import box_cxcywh_to_xyxy, generalied_box_iou
import paddle.nn.functional as F
import paddle.fluid.layers as L
import paddle
import numpy as np
from util import box_ops
from paddle.nn.functional import nll_loss

#多层感知器FFN
class MLP(nn.Layer):
    """ very simple multi-layer perceptron (also called FFN) """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    @paddle.no_grad()
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        
        return x

#匈牙利算法匹配
class HungarianMatcher(nn.Layer):
    """
    This class computes an assigment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """
        Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 erro of the bounding box coordinates in the matching
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        # print("===>", self.cost_class, self.cost_bbox, self.cost_giou)
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"
    
    @paddle.no_grad()
    def forward(self, outputs, targets):
        """
        Performs the matching

        Params:
            outputs: This is a dict contains at least these entries:
                "pred_logits": Tensor of dim[batch_size, num_queries, num_classes] with the classification logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicated box coordinates
            
            targets: This is a list of targets (len(targets) == batch_size), where each target is a dict containing:
                "labels": Tensor of dim[num_target_boxes] (where num_target_boxes is the number of ground-truth)
                          objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordiantes
        
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries, num_classes = outputs["pred_logits"].shape

        # We flatten to compute the cost matrices in a batch
        out_prob = paddle.reshape(outputs["pred_logits"], [-1, num_classes]) # [batch_size * num_queries, num_classes]
        out_prob = F.softmax(out_prob, axis=-1) # [batch_size * num_queries, num_classes]
        out_bbox = paddle.reshape(outputs["pred_boxes"], [-1, 4]) # [batch_size * num_queries, 4]

        # Alse concat the target labels and boxes 
        tgt_ids = paddle.concat([v["labels"] for v in targets]).astype("int64") # [batch_size * num_target_boxes_i]
        tgt_bbox = paddle.concat([v["boxes"] for v in targets]).astype("float32") # [batch_size * num_target_boxes_i]

        # print(tgt_ids, tgt_bbox) #checked
        # exit()

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that donesn't change the matching, it can be ommitted.
        cost_class = -out_prob.numpy()[:, tgt_ids.numpy()] # [batch_size * num_queries, num_all_target_boxes]
        cost_class = paddle.to_tensor(cost_class)

        # print(cost_class) #checked
        # exit()

        # print(out_bbox, tgt_bbox) #100*4, 19*4
        # exit()

        # paddle中没有cdist

        # Compute the L1 cost between boxes
        num_all_target_boxes = tgt_bbox.shape[0]
        # # print("checking...", out_bbox.shape, tgt_bbox.shape, paddle.unsqueeze(out_bbox, [1]).shape, [bs * num_queries, num_all_target_boxes, bs])
        expanded_out_bbox = paddle.expand(paddle.unsqueeze(out_bbox, [1]), [bs * num_queries, num_all_target_boxes, 4]) # [batch_size * num_queries, num_all_target_boxes, 4]
        expanded_tgt_bbox = paddle.expand(paddle.unsqueeze(tgt_bbox, [0]), [bs * num_queries, num_all_target_boxes, 4])     # [batch_size * num_queries, num_all_target_boxes, 4]
        cost_bbox = F.loss.l1_loss(expanded_out_bbox, expanded_tgt_bbox, reduction='none') # [batch_size * num_queries, num_all_target_boxes, 4]
        cost_bbox = L.reduce_sum(cost_bbox, -1) # [batch_size * num_queries, num_all_target_boxes] #坑了好久， 这里是sum不是mean

        # Compute the giou cost between boxes
        cost_giou = - generalied_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = paddle.reshape(C, [bs, num_queries, -1]) # [batch_size, num_queries, num_all_target_boxes]
        # print(C)
        # exit()

        sizes = [len(v["boxes"]) for v in targets]
        
        indices = [linear_sum_assignment(c[i].numpy()) for i, c in enumerate(L.split(C, sizes, dim=-1))]

        return [(paddle.to_tensor(i.astype("int64")), paddle.to_tensor(j.astype("int64")))
                for i, j in indices]



class DETR(nn.Layer):
    """ This is the DETR module that performs object detection """
    
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """
        Initializes the model.

        Parameters:
            backbone: See backbone.py
            transformer: See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie the detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2D(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
    

    def forward(self, samples: NestedTensor):
        """
        The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        
        It returns a dict with following elements:
            - "pred_logits": the classification logits (including no-object) for all queries.
                             Shape = [batch_size x num_queries x (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as 
                            (center_x, center_y, height, width). There values are normalized in [0, 1],
                            relative to the size of each individual image (disregarding possible padding).
                            See PostProcess for information on how to retrive the unnormalized bounding box.
            - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                             dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, paddle.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)


        features, pos = self.backbone(samples)
        return samples
        # src, mask = features[-1].decompose()
        # assert mask is not None
        #
        # hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        #
        #
        # outputs_class = self.class_embed(hs)
        # outputs_coord = F.sigmoid(self.bbox_embed(hs))
        # out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        
        # print("===>out: ", out['pred_logits'])
        # exit()
        #
        # return out
    

    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

#计算损失
class SetCriterion(nn.Layer):
    """
    This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """
        Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special on-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.eos_coef = eos_coef
        empty_weight = paddle.ones([self.num_classes + 1], dtype="float32")
        empty_weight[-1] = self.eos_coef
        self.empty_weight = empty_weight
        # self.add_parameter("empty_weight", empty_weight)
    

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss (NLL)
        targets dict must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        num_classes_plus_1 = outputs["pred_logits"].shape[-1]
        src_logits = outputs["pred_logits"] # [bs, num_queries, num_classes] #OK

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = [t["labels"].numpy()[J.numpy()] for t, (_, J) in zip(targets, indices)]
        target_classes_o = [paddle.to_tensor(t) for t in target_classes_o]
        target_classes_o = paddle.concat(target_classes_o) # [bs * num_object]

        #target_classes_o = paddle.concat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) #to check

        target_classes = paddle.full(shape=src_logits.shape[:2], fill_value=self.num_classes, dtype="int64")



        # target_classes[idx] = target_classes_o 不能直接选取索引进行操作
        idx = np.array([idx[0].numpy(), idx[1].numpy()])
        target_classes = target_classes.numpy()
        target_classes[idx[0], idx[1]] = target_classes_o.numpy()
        target_classes = paddle.to_tensor(target_classes)
        src_logits_trans = paddle.transpose(src_logits, (0, 2, 1))

        #效仿torch： nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
        log_softmax = paddle.nn.LogSoftmax(axis=1)
        log_out = log_softmax(src_logits_trans)
        loss_ce = nll_loss(log_out, target_classes,self.empty_weight)
       
        # target_classes = paddle.unsqueeze(target_classes, axis=2)
        # loss_ce = F.softmax_with_cross_entropy(src_logits, target_classes) # (bs, num_queries, 1)
        # loss_weight = np.ones(loss_ce.shape).astype("float32")
        # loss_weight[(target_classes == self.num_classes).numpy()] = self.eos_coef
        # loss_ce = loss_ce * paddle.to_tensor(loss_weight)
        # loss_ce = L.reduce_mean(loss_ce)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            out_logits = src_logits.numpy()[idx[0], idx[1], :]
            out_logits = paddle.to_tensor(out_logits) # [num_objects, num_classes_plus_1]
            target_labels = paddle.reshape(target_classes_o, (-1, 1))
            target_labels = target_labels.astype("int64")
            # print("----->", out_logits.dtype, target_labels.dtype)
            losses['class_error'] = 100 - 100 * L.accuracy(out_logits, target_labels)
        return losses

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        with paddle.no_grad():
            pred_logits = outputs["pred_logits"] # [bs, num_queries, num_classes]
            tgt_lengths = paddle.to_tensor([len(v["labels"]) for v in targets]).astype("float32")
            # Count the number of predictions that are NOT "no-object" (which is the last class)
            card_pred = L.reduce_sum((L.argmax(pred_logits, -1) != pred_logits.shape[-1] - 1).astype("float32"))
            card_err = F.l1_loss(card_pred, tgt_lengths)
            losses = {"cardinality_error": card_err}
            return losses
    

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        # idx = self._get_src_permutation_idx(indices)
        # src_boxes = outputs["pred_boxes"].numpy()[idx[0].numpy(), idx[1].numpy(), :] # [num_objects, 4]
        # src_boxes = paddle.to_tensor(src_boxes)

        idx = self._get_src_permutation_idx(indices)
        #src_boxes = paddle.index_select(x=outputs['pred_boxes'], index=idx) #选择索引有问题
        src_boxes = outputs["pred_boxes"].numpy()[idx[0].numpy(), idx[1].numpy(), :] # [num_objects, 4]
        src_boxes = paddle.to_tensor(src_boxes)

        #target_boxes = paddle.concat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) #索引问题

        target_boxes = [t["boxes"].numpy()[i.numpy()] for t, (_, i) in zip(targets, indices)]
        target_boxes = [paddle.to_tensor(t) for t in target_boxes]
        target_boxes = paddle.concat(target_boxes, 0).astype("float32") # [num_objects, 4]
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum") 
        losses = {}
        losses["loss_bbox"] = loss_bbox / num_boxes #checked

        loss_giou = 1 - paddle.diag(box_ops.generalied_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))) #有小数点0.002d的差距，貌似可以容忍

        losses['loss_giou'] = loss_giou.sum() / num_boxes # 1.60511994 vs 1.6074
        return losses
    

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # assert "pred_masks" in outputs

        # src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)
        # src_masks = outputs["pred_masks"]
        # src_masks = src_masks[src_idx] # []
        pass

    
    def _get_src_permutation_idx(self, indices):
        # permute prediction following indices
        batch_idx = paddle.concat([paddle.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = paddle.concat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices):
        batch_idx = paddle.cat([paddle.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = paddle.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks
        }
        # assert "masks" != loss, "not implement for mask loss"
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    

    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        Parameters:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                     The expected keys in each dict depends on the losses applied, see each loss'doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}


        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = paddle.to_tensor([num_boxes])

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue 
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses

#后处理
class PostProcess(nn.Layer):
    """ This module converts the model's output into the format expected by the coco api"""
    @paddle.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        # print("====>", out_logits.shape, out_bbox.shape)
        # print("out_bbox", out_bbox)
        # exit()

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        labels = L.argmax(prob[:, :, :-1], axis=-1) # [bs, num_queries]
        scores = L.reduce_max(prob[:, :, :-1], dim=-1)         # [bs, num_queries]
        # print(labels.shape, scores.shape)
        # print(prob.numpy().shape, prob.numpy()[..., :-1].shape)
        # exit()
        # scores, labels = prob[..., :-1]
        # scores, labels = paddle.max(scores, axis=-1)

        # convert to [x0, y0, x1, y1] format
        bs, num_queries, _ = out_bbox.shape
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # print("out_bbox", boxes)
        # exit()
        boxes = paddle.reshape(boxes, (bs, num_queries, 4))
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = paddle.unbind(target_sizes, axis=1)
        # scale_fct = paddle.stack([img_w, img_h, img_w, img_h], axis=1)
        # boxes = boxes * scale_fct[:, None, :] # paddle里不支持这种操作

        scale_fct = paddle.stack([img_w, img_h, img_w, img_h], 1) # [bs, 4]
        # scale_fct = paddle.expand(paddle.unsqueeze(scale_fct, [1]), (scale_fct.shape[0], num_queries, scale_fct.shape[1]))

        boxes = boxes * scale_fct
        # print(boxes.shape)
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


def build_detr_model(args):
    # print("check args: ", args.num_classes)
    # exit()
    backbone = build_backbone(args) #doing....


    transformer = build_deforamble_transformer(args)
    model = DETR(backbone, transformer, num_classes=args.num_classes,
                 num_queries=args.num_queries, aux_loss=args.aux_loss)
    # matcher = HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
    #
    # weight_dict = {"loss_ce": 1, "loss_bbox": args.bbox_loss_coef}
    # weight_dict["loss_giou"] = args.giou_loss_coef
    # if args.aux_loss:
    #     aux_weight_dict = {}
    #     for i in range(args.dec_layers - 1):
    #         aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    #     weight_dict.update(aux_weight_dict)
    # losses = ['labels', 'boxes', 'cardinality']
    # criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict,
    #                         eos_coef=args.eos_coef, losses=losses)
    #
    # postprocessor = PostProcess()
    # return model, criterion, postprocessor
    return model


if __name__ == '__main__':
    #字典转结构体
    class DictToStruct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    #COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    params = {"hidden_dim": 256, "lr_backbone": -1, "masks": False, "dilation": False, "backbone": "resnet50", "num_classes":91, 
        "hidden_dim": 256, "dropout": 0.1, "nheads": 8, "dim_feedforward": 2048, "enc_layers": 6, "dec_layers": 6, "pre_norm": False,
        "num_queries": 100, "aux_loss": True, "set_cost_class": 1, "set_cost_bbox": 5,  "set_cost_giou": 2, "bbox_loss_coef": 5, 
        "giou_loss_coef": 2, "eos_coef":0.1}

    args = DictToStruct(**params)
    model, ctriterion, postprocesser = build_detr_model(args)
    print(model)
    
    



