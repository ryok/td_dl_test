python -m tensorflow-onnx.tf2onnx.convert --saved-model tmp_model_AE --output "AE.onnx"
python -m tensorflow-onnx.tf2onnx.convert --saved-model tmp_model_AE_encoder --output "AE_encoder.onnx"
python -m tensorflow-onnx.tf2onnx.convert --saved-model tmp_model_AE_decoder --output "AE_decoder.onnx"