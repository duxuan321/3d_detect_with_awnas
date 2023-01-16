from .detector3d_template import Detector3DTemplate


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks() #! 这里存的是names
        # self.gene_type = "OrderedDict([('blocks.0.0.out_channels.1', 0.75), ('blocks.0.0.kernel_size', 3),('blocks.0.2.kernel_size', 3), ('blocks.0.4.kernel_size', 3), ('blocks.0.6.kernel_size', 3), ('blocks.1.0.out_channels.1', 1.0), ('blocks.1.0.kernel_size', 3), ('blocks.1.2.kernel_size', 3), ('blocks.1.4.kernel_size', 3), ('blocks.1.6.kernel_size', 3), ('blocks.1.8.kernel_size', 3), ('blocks.1.10.kernel_size', 3), ('blocks.2.0.out_channels.1', 0.25), ('blocks.2.0.kernel_size', 3), ('blocks.2.2.kernel_size', 3), ('blocks.2.4.kernel_size', 3), ('blocks.2.6.kernel_size', 3), ('blocks.2.8.kernel_size', 3), ('blocks.2.10.kernel_size', 3)])"
        self.gene_type = None

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
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
