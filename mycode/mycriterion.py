import logging
import torch
import math
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss
from fairseq.criterions import register_criterion
from fairseq import metrics, utils

@register_criterion('my_label_smoothed_cross_entropy')
class MyLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        sent_threshold,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.sent_threshold = sent_threshold

    def add_args(parser):
        parser.add_argument('--sent-threshold', default=0., type=float,
                            help='threshold for filtering pairs')
        
    def forward(self, model, sample, reduce=True,show=False,update_num=0):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample["net_input"])

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=False)  # loss:(batch*tgt_len, 1)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        src_norm=net_output[-1]["ctx"][-1,:,:,0].transpose(0, 1)  # Batch, Tgt
        tgt_norm=net_output[-1]["ctx"][-1,:,:,1].transpose(0, 1)  # Batch, Tgt

        if update_num>3000:
            tgt_len = sample["target"].ne(self.padding_idx).float().sum(dim=-1)
            with torch.no_grad():
                pos_pen = torch.arange(start=1, end=src_norm.shape[1]+1)**(1/3)
                
                word_r = src_norm/tgt_norm*pos_pen.repeat(src_norm.shape[0],1).cuda()

                sent_r = (word_r * sample["target"].ne(self.padding_idx).float()).sum(dim=-1)/tgt_len
                sent_r = sent_r.gt(self.sent_threshold).unsqueeze(1).repeat(1, src_norm.shape[1])

                final_r = sent_r.view(-1,1).bool()
            loss = (loss*final_r.int()).sum()

        else:
            sent_r = torch.ones(src_norm.shape)
            final_r = torch.ones(loss.shape).bool().cuda()
            loss = loss.sum()


        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.sum().data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "src_norm":src_norm.sum().data,
            "tgt_norm":tgt_norm.sum().data,
            "mask_num": (~final_r & sample["target"].ne(self.padding_idx).view(-1,1)).float().sum().data,
            "left_sent": (sent_r.float().sum(dim=-1)>0).float().sum().data,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        
        if update_num>3000:
            sample_size = (final_r & sample["target"].ne(self.padding_idx).view(-1,1)).float().sum()
            if sample_size==0:
                sample_size=torch.ones(sample_size.shape).float().cuda()
            del src_norm, tgt_norm, sent_r, final_r, pos_pen
        else:
            del sent_r, final_r
        return loss, sample_size, logging_output


    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        LabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)
        
        src_norm_sum = sum(log.get("src_norm", 0) for log in logging_outputs)
        tgt_norm_sum = sum(log.get("tgt_norm", 0) for log in logging_outputs)
        mask_num = sum(log.get("mask_num", 0) for log in logging_outputs)
        left_sent = sum(log.get("left_sent", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        n_sent = sum(log.get("nsentences", 0) for log in logging_outputs)
        metrics.log_scalar(
            "src_norm", src_norm_sum / sample_size, sample_size, round=2
        )
        metrics.log_scalar(
            "tgt_norm", tgt_norm_sum / sample_size, sample_size, round=2
        )
        metrics.log_scalar(
            "mask_rate", mask_num / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "ori_sent", sample_size / n_sent, sample_size, round=2
        )
        metrics.log_scalar(
            "left_sent", (sample_size-mask_num) / left_sent, sample_size, round=2
        )

