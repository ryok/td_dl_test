# Touch Deginer with DL

```bash
docker build . -t `whoami`_td:1.0 -f Dockerfile

docker run --runtime=nvidia -it \
 --rm -v /home/ryo.okada/td_dl_test:/workspace \
 --name `whoami`_td `whoami`_td:1.0 /bin/bash


# git clone https://github.com/onnx/tensorflow-onnx

python ae_test.py

bash tf2onnx_convert.sh
 ```
