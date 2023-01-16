from .detector3d_template import Detector3DTemplate
from icecream import ic

class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks() #! 这里存的是names
        # self.gene_type = "OrderedDict([('blocks.0.0.out_channels.1', 1.0), ('blocks.0.0.kernel_size', 3),('blocks.0.2.kernel_size', 3), ('blocks.0.4.kernel_size', 3), ('blocks.0.6.kernel_size', 3), ('blocks.1.0.out_channels.1', 1.0), ('blocks.1.0.kernel_size', 3), ('blocks.1.2.kernel_size', 3), ('blocks.1.4.kernel_size', 3), ('blocks.1.6.kernel_size', 3), ('blocks.1.8.kernel_size', 3), ('blocks.1.10.kernel_size', 3), ('blocks.2.0.out_channels.1', 1.0), ('blocks.2.0.kernel_size', 3), ('blocks.2.2.kernel_size', 3), ('blocks.2.4.kernel_size', 3), ('blocks.2.6.kernel_size', 3), ('blocks.2.8.kernel_size', 3), ('blocks.2.10.kernel_size', 3)])"
        # self.gene_type = "OrderedDict([('blocks.0.0.out_channels.1', 1.0), ('blocks.0.0.kernel_size', 3), ('blocks.0.2.out_channels.1', 1.0), ('blocks.0.2.kernel_size', 3), ('blocks.0.4.out_channels.1', 1.0), ('blocks.0.4.kernel_size', 3), ('blocks.0.6.kernel_size', 3), ('blocks.1.0.out_channels.1', 1.0), ('blocks.1.0.kernel_size', 3), ('blocks.1.2.out_channels.1', 1.0), ('blocks.1.2.kernel_size', 3), ('blocks.1.4.out_channels.1', 1.0), ('blocks.1.4.kernel_size', 3), ('blocks.1.6.out_channels.1', 1.0), ('blocks.1.6.kernel_size', 3), ('blocks.1.8.out_channels.1', 1.0), ('blocks.1.8.kernel_size', 3), ('blocks.1.10.kernel_size', 3), ('blocks.2.0.out_channels.1', 1.0), ('blocks.2.0.kernel_size', 3), ('blocks.2.2.out_channels.1', 1.0), ('blocks.2.2.kernel_size', 3), ('blocks.2.4.out_channels.1', 1.0), ('blocks.2.4.kernel_size', 3), ('blocks.2.6.out_channels.1', 1.0), ('blocks.2.6.kernel_size', 3), ('blocks.2.8.out_channels.1', 1.0), ('blocks.2.8.kernel_size', 3), ('blocks.2.10.kernel_size', 3)])"        
        self.gene_type = None
        
        self.export_onnx = model_cfg.get("EXPORT_ONNX",False)
        self.with_iou_loss = model_cfg.get("WITH_IOU_LOSS",False)
        self.with_iou_aware_loss = model_cfg.get("WITH_IOU_AWARE_LOSS",False)
        self.iou_weight = model_cfg.get("IOU_WEIGHT",1)
        self.iou_aware_weight = model_cfg.get("IOU_AWARE_WEIGHT",1)


    def forward(self, batch_dict):
        for cur_module in self.module_list:
            cur_module = getattr(self, cur_module)
            if cur_module.__class__.__name__ == 'BEVBackboneSuperNet':
                batch_dict = cur_module(batch_dict, self.gene_type)
            else:
                batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if self.export_onnx:
                return batch_dict
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        if self.with_iou_loss:
            loss_rpn, tb_dict = self.dense_head.get_loss_with_iou(self.iou_weight)
        elif self.with_iou_aware_loss:
            loss_rpn, tb_dict = self.dense_head.get_loss_with_iou_aware(self.iou_aware_weight)
        else:
            loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
