# code referenced from https://github.com/vincentherrmann/pytorch-wavenet/blob/master/model_logging.py
import threading
from io import BytesIO
import time

import tensorflow as tf
import numpy as np
from PIL import Image
import torch


class Accumulator:
    def __init__(self, *keys):
        self._values = {key: 0. for key in keys}
        self.log_interval = 0

    def accumulate(self, **kwargs):
        for key in kwargs:
            self._values[key] += kwargs[key]
        self.log_interval += 1

    def reset(self):
        for key in self._values:
            self._values[key] = 0.
        self.log_interval = 0

    @property
    def values(self):
        return {key: value / self.log_interval for key, value in self._values.items()}

    def __getattr__(self, item):
        return self._values[item] / self.log_interval


class Logger:
    def __init__(self,
                 log_interval=50,
                 validation_interval=200,
                 generate_interval=500,
                 test_interval=None,
                 trainer=None,
                 generate_function=None):
        self.trainer = trainer
        self.log_interval = log_interval
        self.val_interval = validation_interval
        self.gen_interval = generate_interval
        self.test_interval = test_interval
        self.log_time = time.time()
        self.load_time = 0.
        self.accumulator = Accumulator('loss', 'accuracy', 'bitperchar')
        self.generate_function = generate_function
        if self.generate_function is not None:
            self.generate_thread = threading.Thread(target=self.generate_function)
            self.generate_function.daemon = True

    def log(self, current_step, current_losses, current_grad_norm, load_time=0.):
        self.load_time += load_time
        self.accumulator.accumulate(
            loss=float(current_losses['loss'].detach()),
            accuracy=float(current_losses['accuracy'].detach()) if 'accuracy' in current_losses else 0.,
            bitperchar=float(current_losses['bitperchar'].detach()) if 'bitperchar' in current_losses else 0.,
        )

        if current_step % self.log_interval == 0:
            self.log_loss(current_step)
            self.log_time = time.time()
            self.load_time = 0.
            self.accumulator.reset()
        if self.val_interval is not None and self.val_interval > 0 and current_step % self.val_interval == 0:
            self.validate(current_step)
        if self.gen_interval is not None and self.gen_interval > 0 and current_step % self.gen_interval == 0:
            self.generate(current_step)
        if self.test_interval is not None and self.test_interval > 0 and current_step % self.test_interval == 0:
            self.test(current_step)

    def log_loss(self, current_step):
        v = self.accumulator.values
        print(f"{time.time()-self.log_time:6.3f} {self.load_time:6.3f} "
              f"loss, bitperchar, accuracy at step {current_step: 8d}: "
              f"{v['loss']:10.6f}, {v['bitperchar']:10.6f}, {v['accuracy']:10.6f}", flush=True)

    def validate(self, current_step):
        validation = self.trainer.validate()
        if validation is None:
            return
        losses, accuracies, true_outputs, logits, rocs = validation
        print(f"val  losses: {', '.join(['{:6.4f}'.format(loss) for loss in losses])}, "
              f"accuracies: {', '.join(['{:6.2f}%'.format(acc * 100) for acc in accuracies])}, "
              f"logits: {', '.join(['{:6.4f}'.format(logit) for logit in logits])}, "
              f"AUCs: {', '.join(['{:6.4f}'.format(roc) for roc in rocs])}",
              flush=True)

    def generate(self, current_step):
        if self.generate_function is None:
            return

        if self.generate_thread.is_alive():
            print("Last generate is still running, skipping this one")
        else:
            self.generate_thread = threading.Thread(target=self.generate_function, args=[current_step])
            self.generate_thread.daemon = True
            self.generate_thread.start()
    
    def test(self, current_step):
        test = self.trainer.test(num_samples=1)
        if test is None:
            return
        losses, accuracies, true_outputs, logits, rocs = test
        print(f"test losses: {', '.join(['{:6.4f}'.format(loss) for loss in losses])}, "
              f"accuracies: {', '.join(['{:6.2f}%'.format(acc * 100) for acc in accuracies])}, "
              f"logits: {', '.join(['{:6.4f}'.format(logit) for logit in logits])}, "
              f"AUCs: {', '.join(['{:6.4f}'.format(roc) for roc in rocs])}",
              flush=True)


# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
class TensorboardLogger(Logger):
    def __init__(self,
                 log_interval=50,
                 validation_interval=200,
                 generate_interval=500,
                 test_interval=None,
                 trainer=None,
                 generate_function=None,
                 log_dir='logs',
                 log_param_histograms=False,
                 log_image_summaries=True,
                 print_output=False,
                 ):
        super().__init__(log_interval, validation_interval, generate_interval, test_interval,
                         trainer, generate_function)
        self.writer = tf.summary.FileWriter(log_dir)
        self.log_param_histograms = log_param_histograms
        self.log_image_summaries = log_image_summaries
        self.print_output = print_output

    def log(self, current_step, current_losses, current_grad_norm, load_time=0.):
        super(TensorboardLogger, self).log(current_step, current_losses, current_grad_norm, load_time)
        self.scalar_summary('grad norm', current_grad_norm, current_step)
        self.scalar_summary('loss', current_losses['loss'].detach(), current_step)
        if 'accuracy' in current_losses:
            self.scalar_summary('accuracy', current_losses['accuracy'].detach(), current_step)
        if 'bitperchar' in current_losses:
            self.scalar_summary('bitperchar', current_losses['bitperchar'].detach(), current_step)

    def log_loss(self, current_step):
        if self.print_output:
            Logger.log_loss(self, current_step)
        # loss
        avg_loss = self.accumulator.loss
        self.scalar_summary('avg loss', avg_loss, current_step)

        if self.log_param_histograms:
            for tag, value, in self.trainer.model.named_parameters():
                tag = tag.replace('.', '/')
                self.histo_summary(tag, value.data, current_step)
                if value.grad is not None:
                    self.histo_summary(tag + '/grad', value.grad.data, current_step)

        if self.log_image_summaries:
            for tag, summary in self.trainer.model.image_summaries.items():
                self.image_summary(tag, summary['img'], current_step, max_outputs=summary.get('max_outputs', 3))

    def validate(self, current_step):
        validation = self.trainer.validate()
        if validation is None:
            return
        losses, accuracies, true_outputs, logits, rocs = validation
        for i, (loss, acc) in enumerate(zip(losses, accuracies)):
            self.scalar_summary(f'validation loss {i}', loss, current_step)
            self.scalar_summary(f'validation accuracy {i}', acc, current_step)
        if self.print_output:
            print(f"val  losses: {', '.join(['{:6.4f}'.format(loss) for loss in losses])}, "
                  f"accuracies: {', '.join(['{:6.2f}%'.format(acc * 100) for acc in accuracies])}, "
                  f"logits: {', '.join(['{:6.4f}'.format(logit) for logit in logits])}, "
                  f"AUCs: {', '.join(['{:6.4f}'.format(roc) for roc in rocs])}",
                  flush=True)
    
    def test(self, current_step):
        test = self.trainer.test(num_samples=1)
        if test is None:
            return
        losses, accuracies, true_outputs, logits, rocs = test
        for i, (loss, acc) in enumerate(zip(losses, accuracies)):
            self.scalar_summary(f'test loss {i}', loss, current_step)
            self.scalar_summary(f'test accuracy {i}', acc, current_step)
        if self.print_output:
            print(f"test losses: {', '.join(['{:6.4f}'.format(loss) for loss in losses])}, "
                  f"accuracies: {', '.join(['{:6.2f}%'.format(acc * 100) for acc in accuracies])}, "
                  f"logits: {', '.join(['{:6.4f}'.format(logit) for logit in logits])}, "
                  f"AUCs: {', '.join(['{:6.4f}'.format(roc) for roc in rocs])}",
                  flush=True)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if isinstance(value, torch.Tensor):
            value = value.item()  # value must have 1 element only
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step, max_outputs=3):
        """Log a tensor image.
        :param tag: string summary name
        :param images: (N, H, W, C) or (N, H, W)
        :param step: current step
        :param max_outputs: max N images to save
        """

        img_summaries = []
        for i in range(min(images.size(0), max_outputs)):
            img = images[i].cpu().numpy()

            # Write the image to a string
            s = BytesIO()
            Image.fromarray(img, 'RGB').save(s, format="png")
            # scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=200):
        """Log a histogram of the tensor of values."""
        values = values.cpu().numpy()

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def tensor_summary(self, tag, tensor, step):
        tf_tensor = tf.Variable(tensor).to_proto()
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, tensor=tf_tensor)])
        # summary = tf.summary.tensor_summary(name=tag, tensor=tensor)
        self.writer.add_summary(summary, step)
