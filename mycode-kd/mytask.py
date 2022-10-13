import os, torch
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data import data_utils, PrependTokenDataset, LanguagePairDataset, ConcatDataset

# from fairseq import checkpoint_utils
import logging
from fairseq.file_io import PathManager
from torch.serialization import default_restore_location
logger = logging.getLogger(__name__)


def load_teacher_checkpoint_to_cpu(path):
    # with PathManager.open(path, "rb") as f:
    #     state = torch.load(
    #         f, map_location=lambda s, l: default_restore_location(s, "cpu")
    #     )
    with open(PathManager.get_local_path(path), "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    return state 


@register_task('my_translation_task')
class MyTranslationTask(TranslationTask):

    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument('--teacher-ckpt-path', default=None, type=str, help='teacher model')
        pass
    
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.teacher_ckpt_path=args.teacher_ckpt_path
        self.teacher_model=None


    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        
        if self.teacher_ckpt_path is not None and update_num%1000 == 1 and update_num>3000:
            self.teacher_model = self.build_model(model.args)
            teacher_state = load_teacher_checkpoint_to_cpu('%s/checkpoint_best.pt'%self.teacher_ckpt_path)
            self.teacher_model.load_state_dict(
                teacher_state["model"], strict=True, model_cfg=model.args
            )
            logger.info(
            "loaded checkpoint {} ({} updates)".format(
                '%s/checkpoint_best.pt'%self.teacher_ckpt_path, teacher_state["optimizer_history"][-1]['num_updates'],)
            )
            self.teacher_model.to(device=sample["net_input"]['src_tokens'].device)

        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample, update_num=update_num, teacher_model=self.teacher_model)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        # self.teacher_model=None
        return loss, sample_size, logging_output
    


