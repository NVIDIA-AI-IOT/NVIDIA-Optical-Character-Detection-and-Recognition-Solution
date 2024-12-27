dst=onnx_models
mkdir -p $dst

# ocd models
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocdnet/deployable_v1.0/files?redirect=true&path=dcn_resnet18.onnx' \
       -O $dst/dcn_resnet18.onnx

wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocrnet/deployable_v1.0/files?redirect=true&path=ocrnet_resnet50.onnx' \
       -O $dst/ocrnet_resnet50.onnx

wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocdnet/deployable_v2.0/files?redirect=true&path=ocdnet_fan_tiny_2x_icdar.onnx' \
       -O $dst/ocdnet_fan_tiny_2x_icdar.onnx

wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocrnet/deployable_v2.0/files?redirect=true&path=ocrnet-vit.onnx' \
       -O $dst/ocrnet-vit.onnx
