import os
import sys
import json
import paddle
import numpy as np
from pycocotools.coco import COCO

def get_categories(anno_file):
    coco = COCO(anno_file)
    cats = coco.loadCats(coco.getCatIds())
    clsid2catid = {i: cat['id'] for i, cat in enumerate(cats)}
    catid2name = {cat['id']: cat['name'] for cat in cats}
    return clsid2catid, catid2name

def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000),
                 classwise=False):
    """
    Args:
        jsonfile (str): Evaluation json file, eg: bbox.json, mask.json.
        style (str): COCOeval style, can be `bbox` , `segm` and `proposal`.
        coco_gt (str): Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file (str): COCO annotations file.
        max_dets (tuple): COCO evaluation maxDets.
        classwise (bool): Whether per-category AP and draw P-R Curve or not.
    """
    assert coco_gt != None or anno_file != None
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if coco_gt == None:
        coco_gt = COCO(anno_file)
    logger.info("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)
    coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # flush coco evaluation result
    sys.stdout.flush()
    return coco_eval.stats


class Metric(paddle.metric.Metric):
    def name(self):
        return self.__class__.__name__

    def reset(self):
        pass

    def accumulate(self):
        pass

    # paddle.metric.Metric defined :metch:`update`, :meth:`accumulate`
    # :metch:`reset`, in ppdet, we also need following 2 methods:

    # abstract method for logging metric results
    def log(self):
        pass

    # abstract method for getting metric results
    def get_results(self):
        pass


class COCOMetric(Metric):
    def __init__(self, anno_file, **kwargs):
        assert os.path.isfile(anno_file), \
                "anno_file {} not a file".format(anno_file)
        self.anno_file = anno_file
        self.clsid2catid = kwargs.get('clsid2catid', None)
        if self.clsid2catid is None:
            self.clsid2catid, _ = get_categories(anno_file)
        self.classwise = kwargs.get('classwise', False)
        self.output_eval = kwargs.get('output_eval', None)
        # TODO: bias should be unified
        self.bias = kwargs.get('bias', 0)
        self.save_prediction_only = kwargs.get('save_prediction_only', False)
        self.reset()

    def reset(self):
        # only bbox and mask evaluation support currently
        self.results = {'bbox': [], 'mask': [], 'segm': []}
        self.eval_results = {}

    def update(self, inputs, outputs):
        outs = {}

        # print(inputs, outputs)
        # exit()

        # outputs Tensor -> numpy.ndarray
        for k, v in outputs.items():
            outs[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

        # im_id = inputs['im_id']
        # outs['im_id'] = im_id.numpy() if isinstance(im_id,
        #                                             paddle.Tensor) else im_id
        exit()

        
        infer_results = get_infer_results(
            outs, self.clsid2catid, bias=self.bias)
        self.results['bbox'] += infer_results[
            'bbox'] if 'bbox' in infer_results else []


    def accumulate(self):
        if len(self.results['bbox']) > 0:
            output = "bbox.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)

            with open(output, 'w') as f: #保存输出结果到json
                json.dump(self.results['bbox'], f)
                logger.info('The bbox result is saved to bbox.json.')

            if self.save_prediction_only:
                logger.info('The bbox result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                bbox_stats = cocoapi_eval(
                    output,
                    'bbox',
                    anno_file=self.anno_file) 
                self.eval_results['bbox'] = bbox_stats
                sys.stdout.flush()

        

    def log(self):
        pass

    def get_results(self):
        return self.eval_results