import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, shape=None):
        self.host = host_mem
        self.device = device_mem
        self.shape = shape

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TorchTRTModule():
    def __init__(self, engine_path, dtype=torch.float32):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.dtype = dtype
        self.stream = torch.cuda.current_stream().cuda_stream
        self.device = torch.device('cuda')

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def allocate_buffers(self, tensor):
        outputs = []
        binding_names = sorted(self.engine)
        bindings = [None] * len(binding_names)
        for binding in binding_names:
            # Set/get input/output shape
            idx = self.engine.get_binding_index(binding)
            if self.engine.binding_is_input(binding):
                shape = tuple(tensor.shape)
                self.context.set_binding_shape(idx, shape)
                bindings[idx] = tensor.contiguous().data_ptr()
            else:
                shape = tuple(self.context.get_binding_shape(idx))
                output = torch.empty(size=shape, dtype=self.dtype, device=self.device)
                outputs.append(output)
                bindings[idx] = output.data_ptr()

        return outputs, bindings

    def __call__(self, tensor):
        outputs, bindings = self.allocate_buffers(tensor)
        self.context.execute_async_v2(bindings, self.stream)
        outputs = tuple(outputs)
        return outputs
