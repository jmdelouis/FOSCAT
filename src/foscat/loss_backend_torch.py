import torch


class loss_backend:

    def __init__(self, backend, curr_gpu, mpi_rank):

        self.bk = backend
        self.curr_gpu = curr_gpu
        self.mpi_rank = mpi_rank

    def check_dense(self, data, datasz):
        if isinstance(data, torch.Tensor):
            return data
        """
        idx=tf.cast(data.indices, tf.int32)
        data=tf.math.bincount(idx,weights=data.values,
                              minlength=datasz)
        """
        return data

    # ---------------------------------------------−---------

    def loss(self, x, batch, loss_function, KEEP_TRACK):

        operation = loss_function.scat_operator

        if torch.cuda.is_available():
            with torch.cuda.device((operation.gpupos + self.curr_gpu) % operation.ngpu):

                l_x = x.clone().detach().requires_grad_(True)

                if KEEP_TRACK is not None:
                    l_loss, linfo = loss_function.eval(l_x, batch, return_all=True)
                else:
                    l_loss = loss_function.eval(l_x, batch)

                l_loss.backward()

                g = l_x.grad

                self.curr_gpu = self.curr_gpu + 1
        else:
            l_x = x.clone().detach().requires_grad_(True)

            if KEEP_TRACK is not None:
                l_loss, linfo = loss_function.eval(l_x, batch, return_all=True)
            else:
                l_loss = loss_function.eval(l_x, batch)

            l_loss.backward()

            g = l_x.grad

        if KEEP_TRACK is not None:
            return l_loss.detach(), g, linfo
        else:
            return l_loss.detach(), g
