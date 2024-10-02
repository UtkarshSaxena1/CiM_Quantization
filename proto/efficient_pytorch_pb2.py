# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: efficient_pytorch.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17\x65\x66\x66icient_pytorch.proto\x12\x11\x65\x66\x66icient_pytorch\"\xb2\n\n\nHyperParam\x12\x37\n\tmain_file\x18\x01 \x01(\t:$examples/classifier_imagenet/main.py\x12\x15\n\x04\x61rch\x18\x02 \x01(\t:\x07\x61lexnet\x12?\n\x0cmodel_source\x18\x03 \x01(\x0e\x32).efficient_pytorch.HyperParam.ModelSource\x12\x1a\n\x08log_name\x18\x04 \x01(\t:\x08template\x12\x0c\n\x04\x64\x61ta\x18\x05 \x02(\t\x12\r\n\x05\x64\x65\x62ug\x18\x06 \x01(\x08\x12\x14\n\x0coverfit_test\x18\x07 \x01(\x08\x12\x0f\n\x02lr\x18\n \x01(\x02:\x03\x30.1\x12\x12\n\x06\x65pochs\x18\x0b \x01(\x05:\x02\x39\x30\x12\x17\n\nbatch_size\x18\x0c \x01(\x05:\x03\x32\x35\x36\x12\x12\n\x07workers\x18\r \x01(\x05:\x01\x34\x12\x16\n\nprint_freq\x18\x0e \x01(\x05:\x02\x35\x30\x12\x15\n\x08log_freq\x18\xc2\x02 \x01(\x05:\x02\x34\x30\x12\x10\n\x08\x65valuate\x18\x0f \x01(\x08\x12\x12\n\npretrained\x18\x10 \x01(\x08\x12\x1c\n\x13pretrained_location\x18\xc1\x02 \x01(\t\x12\x0c\n\x04seed\x18\x11 \x01(\x05\x12\x13\n\x0b\x65xport_onnx\x18\x12 \x01(\x08\x12\x0e\n\x06resume\x18\x13 \x01(\t\x12\x0e\n\x06weight\x18\x16 \x01(\t\x12&\n\x06gpu_id\x18\x14 \x01(\x0e\x32\x16.efficient_pytorch.GPU\x12.\n\tmulti_gpu\x18\x15 \x01(\x0b\x32\x1b.efficient_pytorch.MultiGPU\x12\'\n\x05qmode\x18\x32 \x01(\x0e\x32\x18.efficient_pytorch.Qmode\x12\x12\n\x07nbits_w\x18\x33 \x01(\x05:\x01\x34\x12\x12\n\x07nbits_a\x18\x34 \x01(\x05:\x01\x34\x12\x17\n\x0bnbits_alpha\x18\x88\x04 \x01(\x05:\x01\x38\x12\x14\n\twbitslice\x18\x35 \x01(\x05:\x01\x31\x12\x14\n\tabitslice\x18\x36 \x01(\x05:\x01\x31\x12\x10\n\x04xbar\x18\x37 \x01(\x05:\x02\x36\x34\x12\x12\n\x07\x61\x64\x63\x62its\x18\x38 \x01(\x02:\x01\x36\x12\x13\n\x0bsigned_xbar\x18: \x01(\x08\x12\x18\n\x10stochastic_quant\x18; \x01(\x08\x12+\n\x07\x63immode\x18\x39 \x01(\x0e\x32\x1a.efficient_pytorch.CimMode\x12)\n\x06warmup\x18\x63 \x01(\x0b\x32\x19.efficient_pytorch.Warmup\x12\x37\n\x0clr_scheduler\x18\x64 \x01(\x0e\x32!.efficient_pytorch.LRScheduleType\x12/\n\x07step_lr\x18\x65 \x01(\x0b\x32\x1e.efficient_pytorch.StepLRParam\x12:\n\rmulti_step_lr\x18\x66 \x01(\x0b\x32#.efficient_pytorch.MultiStepLRParam\x12\x33\n\tcyclic_lr\x18g \x01(\x0b\x32 .efficient_pytorch.CyclicLRParam\x12\x34\n\toptimizer\x18\xc8\x01 \x01(\x0e\x32 .efficient_pytorch.OptimizerType\x12)\n\x03sgd\x18\xc9\x01 \x01(\x0b\x32\x1b.efficient_pytorch.SGDParam\x12+\n\x04\x61\x64\x61m\x18\xca\x01 \x01(\x0b\x32\x1c.efficient_pytorch.AdamParam\"8\n\x0bModelSource\x12\x0f\n\x0bTorchVision\x10\x01\x12\r\n\tPyTorchCV\x10\x02\x12\t\n\x05Local\x10\x03\"\x9d\x01\n\x08MultiGPU\x12\x16\n\nworld_size\x18\x01 \x01(\x05:\x02-1\x12\x0f\n\x04rank\x18\x02 \x01(\x05:\x01\x30\x12\'\n\x08\x64ist_url\x18\x03 \x01(\t:\x15tcp://127.0.0.1:23456\x12\x1a\n\x0c\x64ist_backend\x18\x04 \x01(\t:\x04nccl\x12#\n\x1bmultiprocessing_distributed\x18\x05 \x01(\x08\"?\n\x08SGDParam\x12\x1c\n\x0cweight_decay\x18\x01 \x01(\x02:\x06\x30.0001\x12\x15\n\x08momentum\x18\x02 \x01(\x02:\x03\x30.9\")\n\tAdamParam\x12\x1c\n\x0cweight_decay\x18\x01 \x01(\x02:\x06\x30.0001\"4\n\x06Warmup\x12\x12\n\x06\x65pochs\x18\x01 \x01(\x05:\x02\x31\x30\x12\x16\n\nmultiplier\x18\x02 \x01(\x02:\x02\x31\x30\"8\n\x0bStepLRParam\x12\x15\n\tstep_size\x18\x03 \x01(\x05:\x02\x32\x30\x12\x12\n\x05gamma\x18\x04 \x01(\x02:\x03\x30.1\":\n\x10MultiStepLRParam\x12\x12\n\nmilestones\x18\x03 \x03(\x05\x12\x12\n\x05gamma\x18\x04 \x01(\x02:\x03\x30.1\"\xe3\x01\n\rCyclicLRParam\x12\x0f\n\x07\x62\x61se_lr\x18\x01 \x01(\x02\x12\x0e\n\x06max_lr\x18\x02 \x01(\x02\x12\x1a\n\x0cstep_size_up\x18\x03 \x01(\x05:\x04\x32\x30\x30\x30\x12\x16\n\x0estep_size_down\x18\x04 \x01(\x05\x12\x33\n\x04mode\x18\x05 \x01(\x0e\x32%.efficient_pytorch.CyclicLRParam.Mode\x12\x10\n\x05gamma\x18\x06 \x01(\x02:\x01\x31\"6\n\x04Mode\x12\x0e\n\ntriangular\x10\x01\x12\x0f\n\x0btriangular2\x10\x02\x12\r\n\texp_range\x10\x03*\x18\n\x03GPU\x12\x07\n\x03\x41NY\x10\x01\x12\x08\n\x04NONE\x10\x02*(\n\x05Qmode\x12\x0e\n\nlayer_wise\x10\x01\x12\x0f\n\x0bkernel_wise\x10\x02*(\n\x07\x43imMode\x12\x0f\n\x0b\x63olumn_wise\x10\x01\x12\x0c\n\x08\x62it_wise\x10\x02*\"\n\rOptimizerType\x12\x07\n\x03SGD\x10\x01\x12\x08\n\x04\x41\x64\x61m\x10\x02*R\n\x0eLRScheduleType\x12\n\n\x06StepLR\x10\x01\x12\x0f\n\x0bMultiStepLR\x10\x02\x12\x15\n\x11\x43osineAnnealingLR\x10\x03\x12\x0c\n\x08\x43yclicLR\x10\x04')

_GPU = DESCRIPTOR.enum_types_by_name['GPU']
GPU = enum_type_wrapper.EnumTypeWrapper(_GPU)
_QMODE = DESCRIPTOR.enum_types_by_name['Qmode']
Qmode = enum_type_wrapper.EnumTypeWrapper(_QMODE)
_CIMMODE = DESCRIPTOR.enum_types_by_name['CimMode']
CimMode = enum_type_wrapper.EnumTypeWrapper(_CIMMODE)
_OPTIMIZERTYPE = DESCRIPTOR.enum_types_by_name['OptimizerType']
OptimizerType = enum_type_wrapper.EnumTypeWrapper(_OPTIMIZERTYPE)
_LRSCHEDULETYPE = DESCRIPTOR.enum_types_by_name['LRScheduleType']
LRScheduleType = enum_type_wrapper.EnumTypeWrapper(_LRSCHEDULETYPE)
ANY = 1
NONE = 2
layer_wise = 1
kernel_wise = 2
column_wise = 1
bit_wise = 2
SGD = 1
Adam = 2
StepLR = 1
MultiStepLR = 2
CosineAnnealingLR = 3
CyclicLR = 4


_HYPERPARAM = DESCRIPTOR.message_types_by_name['HyperParam']
_MULTIGPU = DESCRIPTOR.message_types_by_name['MultiGPU']
_SGDPARAM = DESCRIPTOR.message_types_by_name['SGDParam']
_ADAMPARAM = DESCRIPTOR.message_types_by_name['AdamParam']
_WARMUP = DESCRIPTOR.message_types_by_name['Warmup']
_STEPLRPARAM = DESCRIPTOR.message_types_by_name['StepLRParam']
_MULTISTEPLRPARAM = DESCRIPTOR.message_types_by_name['MultiStepLRParam']
_CYCLICLRPARAM = DESCRIPTOR.message_types_by_name['CyclicLRParam']
_HYPERPARAM_MODELSOURCE = _HYPERPARAM.enum_types_by_name['ModelSource']
_CYCLICLRPARAM_MODE = _CYCLICLRPARAM.enum_types_by_name['Mode']
HyperParam = _reflection.GeneratedProtocolMessageType('HyperParam', (_message.Message,), {
  'DESCRIPTOR' : _HYPERPARAM,
  '__module__' : 'efficient_pytorch_pb2'
  # @@protoc_insertion_point(class_scope:efficient_pytorch.HyperParam)
  })
_sym_db.RegisterMessage(HyperParam)

MultiGPU = _reflection.GeneratedProtocolMessageType('MultiGPU', (_message.Message,), {
  'DESCRIPTOR' : _MULTIGPU,
  '__module__' : 'efficient_pytorch_pb2'
  # @@protoc_insertion_point(class_scope:efficient_pytorch.MultiGPU)
  })
_sym_db.RegisterMessage(MultiGPU)

SGDParam = _reflection.GeneratedProtocolMessageType('SGDParam', (_message.Message,), {
  'DESCRIPTOR' : _SGDPARAM,
  '__module__' : 'efficient_pytorch_pb2'
  # @@protoc_insertion_point(class_scope:efficient_pytorch.SGDParam)
  })
_sym_db.RegisterMessage(SGDParam)

AdamParam = _reflection.GeneratedProtocolMessageType('AdamParam', (_message.Message,), {
  'DESCRIPTOR' : _ADAMPARAM,
  '__module__' : 'efficient_pytorch_pb2'
  # @@protoc_insertion_point(class_scope:efficient_pytorch.AdamParam)
  })
_sym_db.RegisterMessage(AdamParam)

Warmup = _reflection.GeneratedProtocolMessageType('Warmup', (_message.Message,), {
  'DESCRIPTOR' : _WARMUP,
  '__module__' : 'efficient_pytorch_pb2'
  # @@protoc_insertion_point(class_scope:efficient_pytorch.Warmup)
  })
_sym_db.RegisterMessage(Warmup)

StepLRParam = _reflection.GeneratedProtocolMessageType('StepLRParam', (_message.Message,), {
  'DESCRIPTOR' : _STEPLRPARAM,
  '__module__' : 'efficient_pytorch_pb2'
  # @@protoc_insertion_point(class_scope:efficient_pytorch.StepLRParam)
  })
_sym_db.RegisterMessage(StepLRParam)

MultiStepLRParam = _reflection.GeneratedProtocolMessageType('MultiStepLRParam', (_message.Message,), {
  'DESCRIPTOR' : _MULTISTEPLRPARAM,
  '__module__' : 'efficient_pytorch_pb2'
  # @@protoc_insertion_point(class_scope:efficient_pytorch.MultiStepLRParam)
  })
_sym_db.RegisterMessage(MultiStepLRParam)

CyclicLRParam = _reflection.GeneratedProtocolMessageType('CyclicLRParam', (_message.Message,), {
  'DESCRIPTOR' : _CYCLICLRPARAM,
  '__module__' : 'efficient_pytorch_pb2'
  # @@protoc_insertion_point(class_scope:efficient_pytorch.CyclicLRParam)
  })
_sym_db.RegisterMessage(CyclicLRParam)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _GPU._serialized_start=2049
  _GPU._serialized_end=2073
  _QMODE._serialized_start=2075
  _QMODE._serialized_end=2115
  _CIMMODE._serialized_start=2117
  _CIMMODE._serialized_end=2157
  _OPTIMIZERTYPE._serialized_start=2159
  _OPTIMIZERTYPE._serialized_end=2193
  _LRSCHEDULETYPE._serialized_start=2195
  _LRSCHEDULETYPE._serialized_end=2277
  _HYPERPARAM._serialized_start=47
  _HYPERPARAM._serialized_end=1377
  _HYPERPARAM_MODELSOURCE._serialized_start=1321
  _HYPERPARAM_MODELSOURCE._serialized_end=1377
  _MULTIGPU._serialized_start=1380
  _MULTIGPU._serialized_end=1537
  _SGDPARAM._serialized_start=1539
  _SGDPARAM._serialized_end=1602
  _ADAMPARAM._serialized_start=1604
  _ADAMPARAM._serialized_end=1645
  _WARMUP._serialized_start=1647
  _WARMUP._serialized_end=1699
  _STEPLRPARAM._serialized_start=1701
  _STEPLRPARAM._serialized_end=1757
  _MULTISTEPLRPARAM._serialized_start=1759
  _MULTISTEPLRPARAM._serialized_end=1817
  _CYCLICLRPARAM._serialized_start=1820
  _CYCLICLRPARAM._serialized_end=2047
  _CYCLICLRPARAM_MODE._serialized_start=1993
  _CYCLICLRPARAM_MODE._serialized_end=2047
# @@protoc_insertion_point(module_scope)
