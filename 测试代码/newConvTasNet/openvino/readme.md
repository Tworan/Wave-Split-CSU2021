# openvino转化Conv-TasNet笔记
- torch模型转换成openvino遵从以下步骤：[torch->onnx->openvino](https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch.html)

## 安装相关包
- [安装onnx](https://pypi.org/project/onnx/)
- 安装onnxruntime

```shell
pip install onnxruntime
```
- [安装openvino](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_pip.html#doxid-openvino-docs-install-guides-installing-openvino-pip)
```shell
pip install openvino
pip install openvino-dev[onnx]
```
## 转换
<details>
<summary>ONNX</summary>
使用torch2onnx.ipynb转换得到模型文件onnx
</details>
<details>
<summary>openvino</summary>
得到模型后,使用如下指令推理得到IR模型

```shell
mo --input_model "model.onnx" --output_dir "output_dir" --input_shape [1,1,32000]
```
</details>