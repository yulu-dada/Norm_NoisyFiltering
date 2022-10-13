import logging
import torch
import math
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss
from fairseq.criterions import register_criterion
from fairseq import metrics, utils
import torch.nn.functional as F

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
    

    def forward(self, model, sample, reduce=True,show=False,update_num=0, teacher_model=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample["net_input"])
        teacher_output = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_output = teacher_model(**sample['net_input'])
                


        loss, nll_loss, KL_loss = self.compute_loss(model, net_output, sample, teacher_output=teacher_output, reduce=False)  # loss:(batch*tgt_len, 1)

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

                sent_r = (word_r * sample["target"].ne(self.padding_idx).float()).sum(dim=-1)/tgt_len  #Batch, 1
                ce_gate = sent_r.gt(self.sent_threshold).unsqueeze(1).repeat(1, src_norm.shape[1]) #Batch, Tgt

                kl_gate = ~ce_gate.view(-1)

            loss = loss[ce_gate.view(-1,1)].sum() + KL_loss[kl_gate].sum() 
            KL_loss=KL_loss[kl_gate].sum()

        else:
            ce_gate = torch.ones(src_norm.shape).bool().cuda()
            loss = loss.sum()
            KL_loss=KL_loss.sum()

        logging_output = {
            "loss": loss.data,
            "kl_loss": KL_loss.data, 
            "nll_loss": nll_loss.sum().data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "src_norm":src_norm.sum().data,
            "tgt_norm":tgt_norm.sum().data,
            "mask_num": (~ce_gate.view(-1,1) & sample["target"].ne(self.padding_idx).view(-1,1)).float().sum().data,
            "left_sent": (ce_gate.float().sum(dim=-1)>0).float().sum().data,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        
        if update_num>3000:
            sample_size = (ce_gate.view(-1,1).bool() & sample["target"].ne(self.padding_idx).view(-1,1)).float().sum()
            if sample_size==0:
                sample_size=torch.ones(sample_size.shape).float().cuda()
            del src_norm, tgt_norm, pos_pen, KL_loss
        else:
            del ce_gate, KL_loss
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, teacher_output=None, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample) # (batch*tgt_len, vocab)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        if teacher_output is not None:
            pad_mask = target.eq(self.padding_idx).view(-1)
            teacher_predict = teacher_output[0]
            teacher_predict = teacher_predict.view(-1, teacher_predict.size(-1)) #Batch*Tgt, vocab
            distil_lprobs = F.softmax(teacher_predict, dim=-1, dtype=torch.float32) #Batch*Tgt, vocab
            KL_loss = F.kl_div(lprobs, distil_lprobs, reduction='none')
            KL_loss = KL_loss.sum(dim=-1) #Batch*Tgt, 1
            KL_loss.masked_fill_(pad_mask, 0.)
        else:
            KL_loss = torch.zeros(loss.shape)
        return loss, nll_loss, KL_loss

    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        LabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)
        
        kl_loss = sum(log.get("kl_loss", 0) for log in logging_outputs)
        src_norm_sum = sum(log.get("src_norm", 0) for log in logging_outputs)
        tgt_norm_sum = sum(log.get("tgt_norm", 0) for log in logging_outputs)
        mask_num = sum(log.get("mask_num", 0) for log in logging_outputs)
        left_sent = sum(log.get("left_sent", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        n_sent = sum(log.get("nsentences", 0) for log in logging_outputs)
        metrics.log_scalar(
            "kl_loss", kl_loss / (mask_num+1), sample_size, round=4
        )
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


