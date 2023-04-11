# lp_recognition_TensorRT

## 1、简介

简介：利用[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)做的一个车牌检测的demo，并且部署到onnx和tensorrt并进行推理。目前只做了**None-VGG-BiLSTM-CTC**这个模型的转换（推理速度快、模型小、准确率不低），后续会看情况更新其他模型。

Requirement：见[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)，此外还需要onnx，onnxruntime，tensorrt-7或8。

train：依照[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)提供的train.py进行训练，需要将多GPU训练改为单GPU训练，见模型修改（1）。

pytorch -> onnx -> tensorrt 部署及推理见4、5。

## 2、模型修改：

（1）因为onnx只支持单GPU部署，原模型是用多GPU训练及推理，所以对这里进行了修改：
```
self.net = torch.nn.DataParallel(self.net).to(self.device) 
```
修改为：
```
self.net = self.net.to(self.device)
```

（2）onnx不支持池化的 **动态参数**，将**Model**中的 `\_\_init\_\_`部分的
```
self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
```
注释掉，将`forward`部分的
```
visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
```
改为
```
avgpool2d = nn.AdaptiveAvgPool2d((1,w))
visual_feature = avgpool2d(visual_feature).permute(0,3,1,2)
```
作者的目的是通过**自适应池化** 将特征图的h维度变为1，保持w维度不变，
再将[b, c, h, w]->[b, w, c, h]，
最后将h维度压缩，->[b, w, c]。
其中w维度作为后面序列建模的长度

（3）其他模型的改动主要是针对单一`None-VGG-BiLSTM-CTC`模型，将其他用不到的参数进行删减，这些都无伤大雅。

## 3、数据集

CCPD2019(4988)+CCPD2020(5769)+CLPD(1200)，共计11957张，训练集：验证集≈8：2。

在RTX 3080上的推理速度（pytorch）：
|模型|单张推理速度/ms|文件权重大小/MB|精度/%|
|:------------------|:--|:---|:---|
|None-VGG-BiLSTM-CTC|2~3|33.9|92.6|
|None-Resnet-BiLSTM-CTC|5~6|188.9|94.7|
|TPS-Resnet-BiLSTM-Attn|8~9|33.9|97.5|

## 4、pytorch -> onnx

利用pytorch自带工具`torch.onnx.export`转换，在`infer_pth.py`中可以实现pytorch模型推理和->onnx的转化.
```
def torch2onnx(model, onnx_path):
        model.eval()
        x = torch.randn(1, 3, 32, 100, device='cuda')
        print(x.shape)
        input_names = ['input']
        output_names = ['output']
        torch.onnx.export(
            model,
            x,
            onnx_path,
            verbose=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=11 
        )
```
onnx的推理在`infer_onnx.py`，onnx模型的导入与加载几乎与pytorch一样。

## 5、onnx -> tensorrt

利用tensorrt自带工具`trtexec`，或者[onnx-tensorrt](https://github.com/onnx/onnx-tensorrt)均可完成。方便起见，这里使用`trtexec`。

首先完成编译：
```
cd <TensorRT root directory>/samples/trtexec
make
```
简单起见，完成fp16精度的转换，（因为使用的时候batch为1，所以这里暂时没有设置成动态形状）：
```
cd <TensorRT root directory>/bin
./trtexec --onnx=model.onnx --workspace=1024 --fp16 
```
注意：tensorrt需要根据不同的GPU进行针对性地优化部署，因此这里提供的.engine文件在你的电脑上大概率用不了，建议按照上面的方式自行转换。

推理过程需要构建engine、context等等，这里借鉴*<TensorRT>/samples/python/efficientnet/infer.py*的写法，具体过程见***infer_trt.py***。

到此完成整个模型的转换及推理，纪念一下～

## 6、Prediction Result 
    
GPU：RTX 3080
|pytorch|onnx|tensorrt|
|:-----------------|:-----------------|:----------------|
|皖DD00507 (0.9733)|皖DD00507 (0.9850)|皖DD00507 (0.9850)|
|皖AD10010 (0.7866)|皖AD10010 (0.9968)|皖AD10010 (0.9969)|
|皖AD86986 (0.6913)|皖AD86986 (0.9804)|皖AD86986 (0.9798)|
|皖BD03960 (0.9078)|皖BD03960 (0.9549)|皖BD03960 (0.9564)|
|皖AD04248 (0.9715)|皖AD04248 (0.9995)|皖AD04248 (0.9995)|
|皖AD09533 (0.9765)|皖AD09533 (0.9806)|皖AD09533 (0.9803)|
|皖AD35169 (0.6677)|皖AD35169 (0.8164)|皖AD35169 (0.8199)|
|皖AD18268 (0.7823)|皖AD18268 (0.9242)|皖AD18268 (0.9249)|
|皖AD12777 (0.5309)|皖AD12777 (0.7267)|皖AD12777 (0.7298)|
|皖AD19889 (0.9881)|皖AD19889 (0.9966)|皖AD19889 (0.9967)|
|皖AD04219 (0.9624)|皖AD04219 (0.9693)|皖AD04219 (0.9693)|
|皖AD02557 (0.7737)|皖AD02557 (0.8962)|皖AD02557 (0.8924)|
|1.37ms|2.69ms|0.95ms|
    
以上推理时间仅为网络的推理时间，不包括前处理及后处理。

## 7、Reference
```
@inproceedings{baek2019STRcomparisons,
  title={What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis},
  author={Baek, Jeonghun and Kim, Geewook and Lee, Junyeop and Park, Sungrae and Han, Dongyoon and Yun, Sangdoo and Oh, Seong Joon and Lee, Hwalsuk},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2019},
  pubstate={published},
  tppubtype={inproceedings}
}
```
## 8、License
> Apache License, Version 2.0
