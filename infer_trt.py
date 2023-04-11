import time

import numpy as np
import cv2
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

import math
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts

class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img

class AlignCollate2(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=True):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        images = batch

        resized_max_w = self.imgW
        input_channel = 3
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.shape[1],image.shape[0]
            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)
            resized_image = cv2.resize(image,(resized_w, self.imgH), cv2.INTER_CUBIC)
            resized_images.append(transform(resized_image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        return image_tensors

class TensorRTInfer:
    
    def __init__(self):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        self.engine_path = 'None-VGG-BiLSTM-CTC.engine'
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0
        
        # process
        self.AlignCollate_demo = AlignCollate2(imgH=32, imgW=100, keep_ratio_with_pad=True)
        self.character = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ藏川鄂甘赣港贵桂黑沪吉冀津晋京辽鲁蒙闽宁青琼陕苏皖湘新渝豫粤云浙'
        self.converter = CTCLabelConverter(self.character)
        

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def infer(self, batch):
        """
        Execute inference on a batch of images(batch_size is 1 in most cases). The images should already be batched and preprocessed, 
        as prepared by the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :return: lp, confidence
        """
        # Prepare the output data
        output = np.zeros(*self.output_spec())

        # Process I/O and execute the network
        batch = self.AlignCollate_demo(batch)    # preprocess
        cuda.memcpy_htod(self.inputs[0]["allocation"], np.ascontiguousarray(batch))
        start = time.time()
        self.context.execute_v2(self.allocations)
        time1 = time.time() - start
        cuda.memcpy_dtoh(output, self.outputs[0]["allocation"])

        # Process the results
        preds = torch.from_numpy(output)
        # Select max probabilty (greedy decoding) then decode index to character
        preds_size = torch.IntTensor([preds.size(1)] * 1)
        _, preds_index = preds.max(2)
        # preds_index = preds_index.view(-1)s
        preds_str = self.converter.decode(preds_index, preds_size)
    
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
            # calculate confidence score (= multiply of pred_max_prob)
        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
         # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]
        

        return pred, confidence_score, time1
    
def warmup():
    # warmup
    for i in range(12):
        img = []
        file_name = "demo_image/" + str(i+1) + ".jpg"
        im = cv2.imread(file_name)
        img.append(im)
        lp, conf, time =recognition.infer(img)

def test():
    avg = 0
    start = time.time()   
    for i in range(12):
        img = []
        file_name = "demo_image/" + str(i+1) + ".jpg"
        im = cv2.imread(file_name)
        img.append(im)
        # start1 = time.time()
        lp, conf, time1 =recognition.infer(img)
        avg = time1 if avg==0 else avg*0.95+time1*0.05
        # print(lp, conf,time.time()-start1)
    print(avg*1000)
    # print((time.time()-start)/12)

if __name__ == '__main__':
    recognition = TensorRTInfer()
    warmup()
    test()